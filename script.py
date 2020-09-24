import pandas as pd
import matplotlib.pyplot as plt


# 2. Roller coasters are thrilling amusement park rides designed to make you squeal and scream! They take you up high,
# drop you to the ground quickly, and sometimes even spin you upside down before returning to a stop. Today you will be
# taking control back from the roller coasters and visualizing data covering international roller coaster rankings and
# roller coaster statistics.
#
# Roller coasters are often split into two main categories based on their construction material: wood or steel. Rankings
# for the best wood and steel roller coasters from the 2013 to 2018 Golden Ticket Awards are provided in
# 'Golden_Ticket_Award_Winners_Wood.csv' and 'Golden_Ticket_Award_Winners_Steel.csv', respectively. Load each csv into
# a DataFrame and inspect it to gain familiarity with the data.

# load rankings data here:

wood_data = pd.read_csv('Golden_Ticket_Award_Winners_Wood.csv')
steel_data = pd.read_csv('Golden_Ticket_Award_Winners_Steel.csv')
print(wood_data.head(), steel_data.head())


# 3. Write a function that will plot the ranking of a given roller coaster over time as a line. Your function should
# take a roller coaster’s name and a ranking DataFrame as arguments. Make sure to include informative labels that
# describe your visualization.
#
# Call your function with "El Toro" as the roller coaster name and the wood ranking DataFrame. What issue do you notice?
# Update your function with an additional argument to alleviate the problem, and retest your function.

# write function to plot rankings over time for 1 roller coaster here:

def ranking_over_time(coaster: str, park: str, rank_data: pd.DataFrame):
    filtered_rankings = rank_data[(rank_data['Name'] == coaster) & (rank_data['Park'] == park)]
    coaster_years = list(filtered_rankings['Year of Rank'])
    coaster_ranks = list(filtered_rankings['Rank'])
    rank_ticks = sorted(list(set(coaster_ranks)))

    ax = plt.subplot()
    plt.plot(coaster_years, coaster_ranks, marker='o')
    plt.xlabel('Year')
    plt.ylabel('Rank')
    plt.title(f'{coaster} Rankings by Year')
    plt.xticks(coaster_years)
    plt.yticks(rank_ticks)
    ax.invert_yaxis()
    plt.subplots_adjust(bottom=0.2)


# plt.subplot(1, 2, 1)
# ranking_over_time('Lightning Racer', 'Hersheypark', wood_data)
# plt.subplot(1, 2, 2)
ranking_over_time('Millennium Force', 'Cedar Point', steel_data)
# plt.show()

plt.clf()

# 4.
# Write a function that will plot the ranking of two given roller coasters over time as lines. Your function should take
# both roller coasters’ names and a ranking DataFrame as arguments. Make sure to include informative labels that
# describe your visualization.
#
# Call your function with "El Toro" as one roller coaster name, “Boulder Dash“ as the other roller coaster name, and the
# wood ranking DataFrame. What issue do you notice? Update your function with two additional arguments to alleviate the
# problem, and retest your function.

# write function to plot rankings over time for 2 roller coasters here:


def two_rankings_over_time(coaster1: str, park1: str, coaster2: str, park2: str, rank_data: pd.DataFrame):
    filtered_rankings1 = rank_data[(rank_data['Name'] == coaster1) & (rank_data['Park'] == park1)]
    filtered_rankings2 = rank_data[(rank_data['Name'] == coaster2) & (rank_data['Park'] == park2)]
    coaster1_years = list(filtered_rankings1['Year of Rank'])
    coaster1_ranks = list(filtered_rankings1['Rank'])
    coaster2_years = list(filtered_rankings2['Year of Rank'])
    coaster2_ranks = list(filtered_rankings2['Rank'])

    rank_ticks = sorted(list(set(coaster1_ranks + coaster2_ranks)))
    year_ticks = sorted(list(set(coaster1_years + coaster2_years)))

    ax = plt.subplot()
    plt.plot(coaster1_years, coaster1_ranks, marker='o', label=coaster1)
    plt.plot(coaster2_years, coaster2_ranks, marker='^', label=coaster2)
    plt.xlabel('Year')
    plt.ylabel('Rank')
    plt.title(f'{coaster1} vs. {coaster2}: Rankings by Year')
    plt.xticks(year_ticks)
    plt.yticks(rank_ticks)
    ax.invert_yaxis()
    plt.legend()
    plt.subplots_adjust(bottom=0.2)


two_rankings_over_time('El Toro', 'Six Flags Great Adventure', 'Boulder Dash', 'Lake Compounce', wood_data)
# plt.show()

plt.clf()

# 5.
# Write a function that will plot the ranking of the top n ranked roller coasters over time as lines. Your function
# should take a number n and a ranking DataFrame as arguments. Make sure to include informative labels that describe
# your visualization.
#
# For example, if n == 5, your function should plot a line for each roller coaster that has a rank of 5 or lower.
#
# Call your function with a value for n and either the wood ranking or steel ranking DataFrame.

# write function to plot top n rankings over time here:


def plot_top_five(n: int, rank_data: pd.DataFrame):
    filtered_data = rank_data[rank_data['Rank'] <= 5]
    coasters = list(filtered_data['Name'].unique())
    years = sorted(list(filtered_data['Year of Rank'].unique()))
    markers = ['s', '*', '^', 'P', 'X']
    marker_idx = 0

    for coaster in coasters:
        coaster_data = filtered_data[filtered_data['Name'] == coaster]
        coaster_ranks = list(coaster_data['Rank'])
        coaster_years = list(coaster_data['Year of Rank'])

        if len(coaster_ranks) < 2:
            marker = markers[marker_idx]
            if marker_idx == 4:
                marker_idx = 0
            else:
                marker_idx += 1
        else:
            marker = '.'

        plt.plot(coaster_years, coaster_ranks, marker=marker, label=coaster)

    ax = plt.subplot()
    plt.xticks(years)
    plt.yticks(range(5, 0, -1))
    plt.xlabel('Years')
    plt.ylabel('Rank')
    plt.title(f'Top Five (in Class) Roller Coasters from {years[0]} to {years[-1]}')
    plt.legend(bbox_to_anchor=(1, 1), fontsize='x-small')
    ax.invert_yaxis()
    plt.subplots_adjust(right=0.75)

    print(filtered_data)


plot_top_five(5, wood_data)
# plt.show()

plt.clf()

# load roller coaster data here:



# write function to plot histogram of column values here:










plt.clf()

# write function to plot inversions by coaster at a park here:










plt.clf()

# write function to plot pie chart of operating status here:










plt.clf()

# write function to create scatter plot of any two numeric columns here:










plt.clf()
