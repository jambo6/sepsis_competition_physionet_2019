import matplotlib.pyplot as plt
import seaborn as sns


def missing_values_barplot(df, missing=True, return_vals=False):
    """Plots the percentage of missing values for each column of the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with some nans.
        missing (bool): Set True to plot percentage missing, False for percentage not missing.
        return_vals (bool): Set True to return the percentages as a pandas series.

    Returns:
        pd.Series: The columns with associated missing percentages.

    """
    # Sorted percentage of missing (or contained) values
    s_amount = 100 * df.isna().sum() / df.shape[0]
    if missing == False:
        s_amount = 100 - s_amount
    s_amount.sort_values(ascending=False, inplace=True)

    # Barplot it
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(s_amount.index, s_amount.values, ax=ax, palette="Blues_d")

    # Some niceties
    ax.set_title('% {} values'.format('Missing' if missing else 'Contained'), weight='bold', fontsize=16)
    plt.xticks(rotation=45)

    if return_vals:
        return s_amount


def remove_plot_ticks(ax, n=5, y_axis=False):
    """ Keeps only every n'th tick on a matplotlib axis """
    axis = ax.xaxis if not y_axis else ax.yaxis
    [l.set_visible(False) for (i, l) in enumerate(axis.get_ticklabels()) if i % n != 0]
