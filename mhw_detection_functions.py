# Authored by Jordan Lorenzatto, UNSW Sydney, Australia
# Contact: j.lorenzatto@unsw.edu.au
# Github: https://github.com/jordanlorenzatto
# Generative AI was used to implement performance and
# formatting improvements to existing code.

import xarray as xr
import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
import warnings
import time


__all__ = ['resample_daily', 'noleap_dayofyear', '_smooth_cyclic', 'make_threshold_specs',
            'compute_climatology_stats', 'build_threshold_climatology', 'apply_dayofyear',
            'prepare_mhw_data', 'select_threshold', 'fill_gaps', 'remove_short_runs',
            'detect_exceedance_events', 'label_exceedance_events', 'check_threshold_nesting',
            '_python_scalar', 'compute_exceedance_event_metrics', 'set_event_multiindex',
            'run_mhw_detection', 'select_events_by_threshold']


##### Calendar and preprocessing helpers #####


def resample_daily(data, var=None):
    """Resample to daily mean and remove 29 February.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Input data with time coord.
    var : str, optional
        Variable to extract if data is Dataset.

    Returns
    -------
    xr.DataArray
        Daily mean data on a 365-day calendar.

    """

    # Extract DataArray if data is Dataset
    if isinstance(data, xr.Dataset):
        if var is None:
            raise ValueError("Must specify 'var' when input is an xr.Dataset.")
        da = data[var]
    elif isinstance(data, xr.DataArray):
        da = data
    else:
        raise TypeError("Input must be an xr.DataArray or xr.Dataset.")

    # Resample to daily and remove Feb 29
    daily = da.resample(time="1D").mean()
    is_feb29 = (daily.time.dt.month == 2) & (daily.time.dt.day == 29)
    return daily.sel(time=~is_feb29)


def noleap_dayofyear(time):
    """Return a 1...365 day-of-year coordinate. 
    Shift dates back if after Feb 28 in a leap year."""

    doy = time.dt.dayofyear

    return xr.where(
        time.dt.is_leap_year & (doy > 59),
        doy - 1,
        doy,
    ).rename("dayofyear")


def _smooth_cyclic(da, smooth_window=31, dim="dayofyear"):
    """Smooth a cyclic coordinate using a centred wrapped rolling mean."""

    if smooth_window % 2 != 1:
        raise ValueError("smooth_window must be odd.")
    pad = smooth_window // 2
    
    return (
        da
        .pad({dim: (pad, pad)}, mode="wrap")
        .rolling({dim: smooth_window}, center=True)
        .mean()
        .isel({dim: slice(pad, -pad)})
        .assign_coords({dim: da[dim]})
    )


##### Threshold specifications #####


def make_threshold_specs(
    quantiles=None,
    offsets=None,
    std_multiples=None,
):
    """Create a table describing requested temperature thresholds for later computation.

    Threshold methods
    -----------------
    quantile
        q_p(t) for value p
    offset
        abs. temperature offset from climatology: climatology(t) + value
    std
        multiples of std dev above climatology: climatology(t) + value * std(t)

    threshold_id in returned DataFrame uniquely identifies combinations of
    threshold method and value. Quantiles and std computed from pooled values.
    """
    rows = []

    # build table of threshold specifications (method, value)
    def add_rows(method, values):
        if values is None:
            return
        vals = np.atleast_1d(values).astype(float)
        for value in vals:
            rows.append({"method": method, "value": float(value)})

    add_rows("quantile", quantiles)
    add_rows("offset", offsets)
    add_rows("std", std_multiples)

    if not rows:
        raise ValueError("At least one threshold must be supplied.")

    specs = pd.DataFrame(rows)

    # check quantiles are in (0,1)
    q = specs.loc[specs["method"] == "quantile", "value"]
    if ((q <= 0) | (q >= 1)).any():
        raise ValueError("Quantile thresholds must lie strictly between 0 and 1.")

    # check for duplicated threshold specifications
    if specs.duplicated(["method", "value"]).any():
        duplicates = specs.loc[
            specs.duplicated(["method", "value"], keep=False),
            ["method", "value"],
        ]
        raise ValueError(f"Duplicate threshold specifications:\n{duplicates}")

    # construct threshold_id identifier
    specs.insert(0, "threshold_id", np.arange(len(specs), dtype=int))

    # add informative label for threshold specifications
    labels = []
    for row in specs.itertuples(index=False):
        if row.method == "quantile":
            labels.append(f"q={row.value:g}")
        elif row.method == "offset":
            labels.append(f"clim+{row.value:g}")
        else:
            labels.append(f"clim+{row.value:g}sigma")
    specs["label"] = labels

    return specs


##### Pooled climatological statistics #####


def compute_climatology_stats(
    baseline_da,
    quantiles=None,
    pool_window=11,
    smooth_window=31,
    min_samples=30,
    std_ddof=1,
):
    """Calculate pooled day-of-year climatological statistics.

    Computes day of year climatology, standard deviation, and quantiles (if 
    requested) from the Hobday-style pooled sample with smoothing.

    Returns
    -------
    xr.Dataset
        including:
        - climatology(dayofyear, ...)
        - std(dayofyear, ...)
        - quantile_threshold(dayofyear, quantile, ...) if requested
    """

    if pool_window % 2 != 1:
        raise ValueError("pool_window must be odd.")
    if smooth_window % 2 != 1:
        raise ValueError("smooth_window must be odd.")

    # check quantile values are in (0,1) if they exist
    if quantiles is None:
        quantiles = np.array([], dtype=float)
    else:
        quantiles = np.unique(np.atleast_1d(quantiles).astype(float))
        if np.any((quantiles <= 0) | (quantiles >= 1)):
            raise ValueError("Quantiles must lie strictly between 0 and 1.")

    # group baseline_da by day of year from 365-day calendar
    doy = noleap_dayofyear(baseline_da.time)
    groups = baseline_da.groupby(doy)

    means = []
    stds = []
    quantile_values = []
    pad = pool_window // 2

    for d in range(1, 366):
        # construct time indices for day of year values to pool
        days = ((np.arange(d - pad, d + pad + 1) - 1) % 365) + 1
        available_days = [day for day in days if day in groups.groups]
        if not available_days:
            raise ValueError(f"No baseline observations available around day-of-year {d}.")

        # construct pooled sample using time indices
        sample = xr.concat([groups[day] for day in available_days], dim="time")
        sample = sample.dropna(dim="time", how="all")
        valid_count = sample.count("time")

        # compute mean and std from pooled sample
        means.append(
            sample.mean("time", skipna=True).where(valid_count >= min_samples)
        )
        stds.append(
            sample.std("time", skipna=True, ddof=std_ddof).where(
                valid_count >= min_samples
            )
        )

        # compute specified quantiles from pooled sample
        if quantiles.size:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=RuntimeWarning,
                    message=".*All-NaN slice encountered.*",
                )
                q = sample.quantile(quantiles, dim="time", skipna=True)
            quantile_values.append(q.where(valid_count >= min_samples))

    # compute cyclic and smoothed climatological mean and std
    dayofyear = np.arange(1, 366)
    climatology = xr.concat(means, dim="dayofyear").assign_coords(dayofyear=dayofyear)
    std = xr.concat(stds, dim="dayofyear").assign_coords(dayofyear=dayofyear)
    
    output = xr.Dataset(
        {
            "climatology": _smooth_cyclic(climatology, smooth_window),
            "std": _smooth_cyclic(std, smooth_window),
        }
    )

    # compute cyclic and smoothed climatological quantiles
    if quantiles.size:
        q = xr.concat(quantile_values, dim="dayofyear").assign_coords(dayofyear=dayofyear)
        output["quantile_threshold"] = _smooth_cyclic(q, smooth_window)

    return output


##### Build and map thresholds #####


def build_threshold_climatology(stats, threshold_specs, atol=1e-12):
    """Construct one threshold DataArray with threshold_id as a dimension.
    Computes actual temperature thresholds based on threshold_specs table."""
    
    # verify threshold_specs is consistent with make_threshold_specs output
    specs = threshold_specs.copy().reset_index(drop=True)
    required = {"threshold_id", "method", "value"}
    missing = required.difference(specs.columns)
    if missing:
        raise ValueError(f"threshold_specs is missing columns: {sorted(missing)}")

    thresholds = []

    # iterate through each threshold specification tuple
    for row in specs.itertuples(index=False):
        method = row.method
        value = float(row.value)

        # check that quantiles from stats match quantiles from specs
        if method == "quantile":
            if "quantile_threshold" not in stats:
                raise ValueError("Quantile thresholds were not computed in stats.")
            available = np.asarray(stats["quantile"].values, dtype=float)
            matches = np.flatnonzero(np.isclose(available, value, rtol=0.0, atol=atol))
            if matches.size != 1:
                raise ValueError(
                    f"Could not uniquely match quantile {value}. "
                    f"Available quantiles are {available}."
                )
            # compute temperature threshold based on quantile spec
            threshold = stats["quantile_threshold"].isel(
                quantile=int(matches[0]), drop=True
            )

        # compute temperature threshold based on offset spec
        elif method == "offset":
            threshold = stats["climatology"] + value

        # compute temperature threshold based on std spec
        elif method == "std":
            threshold = stats["climatology"] + value * stats["std"]

        else:
            raise ValueError(f"Unknown threshold method: {method!r}")

        # add computed threshold to list of thresholds
        thresholds.append(
            threshold.expand_dims(threshold_id=[int(row.threshold_id)])
        )

    # construct DataArray from list of computed thresholds
    out = xr.concat(thresholds, dim="threshold_id").rename("threshold")
    out = out.assign_coords(
        threshold_method=("threshold_id", specs["method"].astype(str).to_numpy()),
        threshold_value=("threshold_id", specs["value"].astype(float).to_numpy()),
        threshold_label=("threshold_id", specs["label"].astype(str).to_numpy()),
    )
    return out


def apply_dayofyear(obj, daily):
    """Map a day-of-year DataArray/Dataset onto the time axis of daily."""
    doy = noleap_dayofyear(daily.time)
    return obj.sel(dayofyear=doy).assign_coords(time=daily.time)


def prepare_mhw_data(daily, stats, threshold_climatology):
    """Combine daily SST, climatology, standard deviation, thresholds, 
    and temperature anomalies (computed herein) into a single Dataset."""
    
    # put climatology, std and thresholds onto day of year time coord
    climatology = apply_dayofyear(stats["climatology"], daily)
    std = apply_dayofyear(stats["std"], daily)
    threshold = apply_dayofyear(threshold_climatology, daily)

    # construct output dataset
    output = xr.Dataset(
        {
            "temp": daily,
            "climatology": climatology,
            "std": std,
            "threshold": threshold,
        }
    )
    
    # add temperature anomalies to output dataset
    output["temp_anomaly"] = output["temp"] - output["climatology"]

    return output


def select_threshold(da, method, value, atol=1e-12):
    """Extract a threshold from output of run_mhw_detection
    by its metadata (method name and value)."""
    
    # extract methods and values from run_mhw_detection output
    methods = np.asarray(da["threshold_method"].values).astype(str)
    values = np.asarray(da["threshold_value"].values, dtype=float)
    matches = np.flatnonzero(
        (methods == method) & np.isclose(values, float(value), rtol=0.0, atol=atol)
    )

    # check for valid and non-coinciding threshold values
    if matches.size != 1:
        raise KeyError(
            f"Expected one threshold matching method={method!r}, value={value}; "
            f"found {matches.size}."
        )
    
    return da.isel(threshold_id=int(matches[0]), drop=True)


##### Event detection #####


def fill_gaps(mask, max_gap_length=2):
    """Fill short False gaps that are bounded by True values."""

    # uniquely identify runs of False values
    mask = np.asarray(mask, dtype=bool)
    output = mask.copy()
    gap_labels, n_gaps = ndimage.label(~mask)

    # iterate through each run of False values
    for gap_label in range(1, n_gaps + 1):
        gap_indices = np.where(gap_labels == gap_label)[0]
        gap_start, gap_end = gap_indices[0], gap_indices[-1]
        gap_length = len(gap_indices)
        
        # fill runs of length <= max_gap_length with True
        if (
            gap_length <= max_gap_length
            and gap_start > 0
            and gap_end < len(mask) - 1
            and mask[gap_start - 1]
            and mask[gap_end + 1]
        ):
            output[gap_indices] = True

    return output


def remove_short_runs(mask, min_run_length=5):
    """Remove True runs shorter than min_run_length."""

    # uniquely identify runs of True values
    mask = np.asarray(mask, dtype=bool)
    output = mask.copy()
    run_labels, n_runs = ndimage.label(mask)

    # iterate through each run of True values
    for run_label in range(1, n_runs + 1):
        run = run_labels == run_label

        # fill runs of length < min_run_length with False
        if run.sum() < min_run_length:
            output[run] = False

    return output


def detect_exceedance_events(exceedance, max_gap=2, min_duration=5):
    """Apply Hobday-style gap joining and minimum-duration filtering
    of exceedance events."""

    # apply fill_gaps to exceedance dataset
    filled = xr.apply_ufunc(
        fill_gaps,
        exceedance,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        kwargs={"max_gap_length": max_gap},
        output_dtypes=[bool],
    )

    # apply remove_short_runs to gap-filled exceedance dataset
    events = xr.apply_ufunc(
        remove_short_runs,
        filled,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        kwargs={"min_run_length": min_duration},
        output_dtypes=[bool],
    )

    return events.rename("event_mask")


def label_exceedance_events(event_mask):
    """Label contiguous event runs independently for every non-time coordinate."""

    # uniquely identify exceedance events
    def _label(mask):
        labels, _ = ndimage.label(mask)
        return labels.astype(np.int32)

    # add exceedance event identifier
    return xr.apply_ufunc(
        _label,
        event_mask,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        output_dtypes=[np.int32],
    ).rename("event_labels")


##### Threshold-nesting validation #####
# For a fixed threshold type, increasing the threshold value must not create 
# exceedance days that were absent at a lower threshold. The same nesting property 
# should remain true after gap joining and the minimum-duration filter. 
# These checks make it possible to distinguish a threshold/alignment bug from 
# a later event-catalogue or plotting bug.


def check_threshold_nesting(mask, raise_on_error=True):
    """Check that increasing thresholds never create longer exceedance events
    than those detected at lower thresholds.

    Nesting is tested separately within each threshold method. For quantiles,
    offsets, and standard-deviation multiples, increasing threshold_value
    corresponds to an equal or higher absolute temperature threshold.
    """

    # check that exceedance event mask contains valid threshold metadata
    required = {"threshold_id", "threshold_method", "threshold_value"}
    if not required.issubset(mask.coords):
        missing = sorted(required.difference(mask.coords))
        raise ValueError(f"Mask is missing threshold coordinates: {missing}")

    rows = []
    methods = pd.unique(np.asarray(mask["threshold_method"].values).astype(str))

    # iterate through all instances of each method used, sorted by threshold_value
    for method in methods:
        ids = np.flatnonzero(
            np.asarray(mask["threshold_method"].values).astype(str) == method
        )
        subset = mask.isel(threshold_id=ids)
        order = np.argsort(np.asarray(subset["threshold_value"].values, dtype=float))
        subset = subset.isel(threshold_id=order)
        values = np.asarray(subset["threshold_value"].values, dtype=float)

        # check for nesting violations where an increasing threshold produces
        # more exceedance days than a lower threshold
        if subset.sizes["threshold_id"] < 2:
            n_violations = 0
        else:
            gained_true = subset.astype(np.int8).diff("threshold_id") > 0
            total = gained_true.sum()
            if hasattr(total.data, "compute"):
                total = total.compute()
            n_violations = int(total.item())

        rows.append(
            {
                "threshold_method": method,
                "n_thresholds": int(subset.sizes["threshold_id"]),
                "min_value": float(values.min()),
                "max_value": float(values.max()),
                "n_violations": n_violations,
                "is_nested": n_violations == 0,
            }
        )

    # raise error if nesting violations occur and list violating threshold specs
    report = pd.DataFrame(rows)
    if raise_on_error and not report["is_nested"].all():
        failed = report.loc[~report["is_nested"]]
        raise AssertionError(
            "Threshold nesting failed. A higher threshold contains True values "
            "where a lower threshold does not.\n" + failed.to_string(index=False)
        )
    return report


##### Event metrics and multi-index event catalogiue #####


def _python_scalar(value):
    """Convert a NumPy scalar to a Python scalar when possible."""
    return value.item() if hasattr(value, "item") else value


def compute_exceedance_event_metrics(
    data,
    event_labels,
    anomaly_var="temp_anomaly",
):
    """Compute Hobday-style event metrics for every threshold/spatial slice.

    event_id is local to each threshold and spatial coordinate. It is not 
    intended to match the same physical event across different thresholds.
    """

    anomaly = data[anomaly_var]
    non_time_dims = [dim for dim in event_labels.dims if dim != "time"]
    sizes = [event_labels.sizes[dim] for dim in non_time_dims]

    # iterate through exceedance events for each non-time dim
    iterator = np.ndindex(*sizes) if sizes else [()]
    rows = []

    for index in iterator:

        isel = dict(zip(non_time_dims, index))
        labels = event_labels.isel(isel)

        selector = {}
        for dim, i in isel.items():
            if dim in event_labels.coords:
                selector[dim] = _python_scalar(event_labels[dim].values[i])
            else:
                selector[dim] = int(i)

        # select anomalies for each exceedance event
        anomaly_isel = {dim: i for dim, i in isel.items() if dim in anomaly.dims}
        anom = anomaly.isel(anomaly_isel)

        # add id and method to selector for each threshold spec
        if "threshold_id" in isel:
            ti = isel["threshold_id"]
            selector["threshold_method"] = str(
                event_labels["threshold_method"].values[ti]
            )
            selector["threshold_value"] = float(
                event_labels["threshold_value"].values[ti]
            )
            if "threshold_label" in event_labels.coords:
                selector["threshold_label"] = str(
                    event_labels["threshold_label"].values[ti]
                )

        # compute number of exceedance events to iterate through
        labels_np = np.asarray(labels.values)
        n_events = int(labels_np.max()) if labels_np.size else 0

        # iterate through all exceedance events and compute events statistics
        for event_id in range(1, n_events + 1):
            event_data = anom.where(labels == event_id, drop=True)
            if event_data.sizes.get("time", 0) == 0:
                continue
            
            # select anomalies for current event
            values = np.asarray(event_data.values)

            # start date, end date, duration
            start_date = event_data.time.values[0]
            end_date = event_data.time.values[-1]
            duration = int(event_data.sizes["time"])

            # max, mean and cumulative intensity (anomaly), 
            # and date of maximum intensity
            max_ind = int(np.nanargmax(values))
            max_i = float(np.nanmax(values))
            mean_i = float(np.nanmean(values))
            cum_i = float(np.nansum(values))
            max_date = event_data.time.values[max_ind]

            # onset and decline rates
            onset_dt = max_ind
            decline_dt = duration - 1 - max_ind
            onset_dvar = max_i - float(values[0])
            decline_dvar = max_i - float(values[-1])
            rate_onset = onset_dvar / onset_dt if onset_dt > 0 else np.nan
            rate_decline = decline_dvar / decline_dt if decline_dt > 0 else np.nan

            rows.append(
                {
                    **selector,
                    "event_id": int(event_id),
                    "start_date": start_date,
                    "end_date": end_date,
                    "duration": duration,
                    "max_intensity": max_i,
                    "max_date": max_date,
                    "mean_intensity": mean_i,
                    "cumulative_intensity": cum_i,
                    "rate_onset": rate_onset,
                    "rate_decline": rate_decline,
                }
            )

    return pd.DataFrame(rows)


def set_event_multiindex(events, spatial_dims=("depth",)):
    """Index an event catalogue by threshold metadata, spatial position, and event id."""
    
    # if no exceedance events, return empty DataFrame
    if events.empty:
        return events
    
    # set multi-index of exceedance event catalogue
    index_cols = ["threshold_method", "threshold_value"]
    index_cols += [dim for dim in spatial_dims if dim in events.columns]
    index_cols += ["event_id"]
    return events.set_index(index_cols).sort_index()


def select_events_by_threshold(catalogue, threshold_method, threshold_value, atol=1e-8):
    """Subset an event catalogue by threshold method and value."""
    
    subset = catalogue.xs(threshold_method, level="threshold_method")

    values = subset.index.get_level_values("threshold_value")
    mask = np.isclose(values, threshold_value, atol=atol, rtol=0)

    if not mask.any():
        raise KeyError(
            f"No {method} threshold close to {threshold_value} "
            f"within atol={atol}."
        )

    return subset[mask]


##### Whole MHW detection and event catalogue compiatlion workflow #####


def run_mhw_detection(
    daily,
    threshold_specs,
    baseline=None,
    pool_window=11,
    smooth_window=31,
    min_samples=30,
    std_ddof=1,
    max_gap=2,
    min_duration=5,
    validate_nesting=True,
    multiindex=True,
):
    """Run the complete threshold -> event-mask -> event-catalogue workflow."""

    # set baseline period; default to entire daily input temperature dataset
    if baseline is None:
        baseline = daily

    # extract threshold_specs and quantiles to compute
    specs = threshold_specs.copy().reset_index(drop=True)
    quantiles = specs.loc[specs["method"] == "quantile", "value"].to_numpy()

    # compute climatological stats, including quantiles if specified
    stats = compute_climatology_stats(
        baseline,
        quantiles=quantiles,
        pool_window=pool_window,
        smooth_window=smooth_window,
        min_samples=min_samples,
        std_ddof=std_ddof,
    )

    # compute thresholds and compile into single dataset
    threshold_climatology = build_threshold_climatology(stats, specs)
    mhw_data = prepare_mhw_data(daily, stats, threshold_climatology)

    # compute exceedance events by broadcasting over all threshold sub-coordinates
    # and check nesting criteria
    exceedance = (mhw_data["temp"] > mhw_data["threshold"]).rename("exceedance")
    raw_nesting = check_threshold_nesting(
        exceedance,
        raise_on_error=validate_nesting,
    )

    # create event mask with Hobday-style gap filling
    # and minimum-duration filtering of exceedance events
    event_mask = detect_exceedance_events(
        exceedance,
        max_gap=max_gap,
        min_duration=min_duration,
    )

    # check nesting criteria again
    event_nesting = check_threshold_nesting(
        event_mask,
        raise_on_error=validate_nesting,
    )

    # get event labels and multi-index event catalogue for return
    event_labels = label_exceedance_events(event_mask)
    events = compute_exceedance_event_metrics(mhw_data, event_labels)
    if multiindex and not events.empty:
        spatial_dims = tuple(dim for dim in daily.dims if dim != "time")
        events = set_event_multiindex(events, spatial_dims=spatial_dims)

    return {
        "threshold_specs": specs,
        "climatology_stats": stats,
        "threshold_climatology": threshold_climatology,
        "data": mhw_data,
        "exceedance": exceedance,
        "event_mask": event_mask,
        "event_labels": event_labels,
        "raw_nesting": raw_nesting,
        "event_nesting": event_nesting,
        "events": events,
    }


def select_exceedance_events(catalogue, threshold_method, threshold_value, atol=1e-8):
    
    subset = catalogue.xs(threshold_method, level="threshold_method")

    values = subset.index.get_level_values("threshold_value")
    mask = np.isclose(values, threshold_value, atol=atol, rtol=0)

    if not mask.any():
        raise KeyError(
            f"No {method} threshold close to {threshold_value} "
            f"within atol={atol}."
        )

    return subset[mask]