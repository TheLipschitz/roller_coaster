import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import inflect
p = inflect.engine()


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
# print(wood_data.head(), steel_data.head())


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


def plot_top_n(n: int, rank_data: pd.DataFrame):
    filtered_data = rank_data[rank_data['Rank'] <= n]
    coasters = list(filtered_data['Name'].unique())
    years = sorted(list(filtered_data['Year of Rank'].unique()))
    markers = ['s', '*', '^', 'P', 'X']
    marker_idx = 0
    number_word = p.number_to_words(n)

    for coaster in coasters:
        coaster_data = filtered_data[filtered_data['Name'] == coaster]
        coaster_ranks = list(coaster_data['Rank'])
        coaster_years = list(coaster_data['Year of Rank'])

        if len(coaster_ranks) < 3:
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
    plt.yticks(range(n, 0, -1))
    plt.xlabel('Years')
    plt.ylabel('Rank')
    plt.title(f'Top {number_word.title()} (in Class) Roller Coasters from {years[0]} to {years[-1]}')
    plt.legend(bbox_to_anchor=(1, 1), fontsize='x-small')
    ax.invert_yaxis()
    plt.subplots_adjust(right=0.75)


plot_top_n(5, wood_data)
# plt.show()

plt.clf()

# 6.
# Now that you’ve visualized rankings over time, let’s dive into the actual statistics of roller coasters themselves.
# Captain Coaster is a popular site for recording roller coaster information. Data on all roller coasters documented on
# Captain Coaster has been accessed through its API and stored in roller_coasters.csv. Load the data from the csv into
# a DataFrame and inspect it to gain familiarity with the data.
#
# Open the hint for more information about each column of the dataset.

# load roller coaster data here:

coaster_data = pd.read_csv('roller_coasters.csv')
# print(coaster_data.head())

# 7.
# Write a function that plots a histogram of any numeric column of the roller coaster DataFrame. Your function
# should take a DataFrame and a column name for which a histogram should be constructed as arguments. Make sure
# to include informative labels that describe your visualization.
#
# Call your function with the roller coaster DataFrame and one of the column names.
# write function to plot histogram of column values here:


def column_hist(data: pd.DataFrame, column_name: str):
    accepted_columns = ['speed', 'height', 'length', 'num_inversions']
    if column_name not in accepted_columns:
        print(f'The column "{column_name}" doesn\'t contain numerical data, please try a different column.')
        return

    if column_name == accepted_columns[0]:
        x_label = 'Speed in Kilometers per Hour'
    elif column_name == accepted_columns[3]:
        x_label = 'Number of Inversions'
    else:
        x_label = f'{column_name.title()} in Meters'

    ax = plt.subplot()
    column_values = list(data[column_name].dropna())
    max_val = data[column_name].max()
    min_val = data[column_name].min()
    bin_count = max(10, min(20, int(max_val - min_val)))

    counts, bins = plt.hist(column_values, bins=bin_count, edgecolor='lightblue')[:2]
    plt.xlabel(x_label)
    plt.xticks([max(bins[i], 0) for i in range(len(bins)) if i % 2 == 0], fontsize='small')
    plt.yticks(fontsize='small')
    bin_centers = 0.5 * np.diff(bins) + bins[:-1]
    for count, x in zip(counts, bin_centers):
        ax.annotate(str(int(count)), xy=(x, 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, 6), textcoords='offset points', va='top', ha='center', fontsize='xx-small')
    plt.ylabel('Roller Coasters')
    plt.title(f'Distribution of {x_label}')
    plt.subplots_adjust(bottom=0.15)


column_hist(coaster_data, 'speed')
# plt.show()

plt.clf()

# 8.
# Write a function that creates a bar chart showing the number of inversions for each roller coaster at an amusement
# park. Your function should take the roller coaster DataFrame and an amusement park name as arguments. Make sure to
# include informative labels that describe your visualization.
#
# Call your function with the roller coaster DataFrame and an amusement park name.
#
# write function to plot inversions by coaster at a park here:


def inversions_by_park(data: pd.DataFrame, park_name: str):
    if park_name not in list(data.park):
        print(f'{park_name} is not a park found in the dataset.')
        return
    park_data = data[(data.park == park_name) & (data.status == 'status.operating')].sort_values('name').fillna(0)
    coaster_names = list(park_data.name)
    if not coaster_names:
        print(f'{park_name} does not have any operating roller coasters.')
        return
    inversion_counts = park_data.num_inversions
    x_ticks = range(len(coaster_names))
    y_ticks = range(int(inversion_counts.max()) + 1)
    if len(x_ticks) > 15:
        f_size = 'x-small'
    elif len(x_ticks) > 10:
        f_size = 'small'
    else:
        f_size = 'medium'

    ax = plt.subplot()
    plt.bar(x_ticks, inversion_counts)
    plt.xticks(x_ticks, rotation=50, fontsize=f_size, ha='right', y=0.01)
    ax.set_xticklabels(coaster_names)
    plt.yticks(y_ticks)
    plt.ylabel('Number of Inversions')
    plt.subplots_adjust(bottom=0.3)
    plt.title(f'Inversions by Roller Coaster at {park_name}')

# This code was used to test a variety of inputs all at once:

# operating_coasters = coaster_data[coaster_data.status == 'status.operating']
# data_by_park = operating_coasters.groupby('park').name.count().reset_index()
# parks_over_five = data_by_park[data_by_park.name >= 5].park
#
# for park in parks_over_five:
#     inversions_by_park(coaster_data, park)
#     plt.show()

inversions_by_park(coaster_data, 'Cedar Point')
# plt.show()

plt.clf()

# 9.
# Write a function that creates a pie chart that compares the number of operating roller coasters ('status.operating')
# to the number of closed roller coasters ('status.closed.definitely'). Your function should take the roller coaster
# DataFrame as an argument. Make sure to include informative labels that describe your visualization.
#
# Call your function with the roller coaster DataFrame.
#
# write function to plot pie chart of operating status here:


def operating_pie(data: pd.DataFrame, park=None):
    if park:
        park_name = park
        data = data[data.park == park]
    else:
        park_name = 'All Parks'

    operating = data[data.status == 'status.operating'].status.count()
    closed = data[data.status == 'status.closed.definitely'].status.count()
    # print(operating, closed, operating + closed, data.count()
    all_coasters = [operating, closed]

    autotexts = plt.pie(all_coasters, colors=['green', 'red'], autopct='%.1f%%')[2]
    plt.legend(bbox_to_anchor=(0.95, 0.95), labels=['Operating', 'Closed'])

    for i, text in enumerate(autotexts):
        percent = text.get_text()
        if percent == '0.0%':
            formatted = ''
        else:
            formatted = '\n'.join([percent, f'({all_coasters[i]})'])
        text.set_text(formatted)
    if operating == 0:
        autotexts[1].set_position((0, 0))
    if closed == 0:
        autotexts[0].set_position((0, 0))
    plt.title(f'Operating vs. Closed Roller Coasters at {park_name}')
    plt.axis('equal')


operating_pie(coaster_data)
# plt.show()

plt.clf()

# 10.
# .scatter() is another useful function in matplotlib that you might not have seen before. .scatter() produces a scatter
# plot, which is similar to .plot() in that it plots points on a figure. .scatter(), however, does not connect the
# points with a line. This allows you to analyze the relationship between to variables.
#
# Write a function that creates a scatter plot of two numeric columns of the roller coaster DataFrame. Your function
# should take the roller coaster DataFrame and two-column names as arguments. Make sure to include informative labels
# that describe your visualization.
#
# Call your function with the roller coaster DataFrame and two-column names.

# write function to create scatter plot of any two numeric columns here:

def make_a_scatter(data: pd.DataFrame, column1_name: str, column2_name: str):
    accepted_columns = ['speed', 'height', 'length', 'num_inversions']
    if column1_name not in accepted_columns:
        print(f'The column "{column1_name}" doesn\'t contain numerical data, please try a different column.')
        return
    if column2_name not in accepted_columns:
        print(f'The column "{column2_name}" doesn\'t contain numerical data, please try a different column.')
        return

    column1 = list(data[column1_name])
    column2 = list(data[column2_name])

    if column1_name == 'num_inversions':
        column1_name = 'Number of Inversions'
    else:
        column1_name = column1_name.title()
        if column2_name == 'num_inversions':
            column2_name = 'Number of Inversions'
        else:
            column2_name = column1_name.title()

    plt.scatter(column1, column2, marker='x')
    plt.xlabel(column1_name)
    plt.ylabel(column2_name)
    plt.title(f'{column2_name} vs. {column1_name}')


make_a_scatter(coaster_data, 'speed', 'num_inversions')
# plt.show()

plt.clf()

# 11.
# Part of the fun of data analysis and visualization is digging into the data you have and answering questions
# that come to your mind.
#
# Some questions you might want to answer with the datasets provided include:
#
# What roller coaster seating type is most popular? And do different seating types result in higher/faster/longer
# roller coasters?


def seating_popularity(data: pd.DataFrame):
    filtered = data[(data.seating_type != 'Sit Down') & (data.seating_type != 'na')].dropna().reset_index()
    seat_data = filtered.groupby(filtered.seating_type).name.count().reset_index()
    seat_data.rename(columns={'name': 'counts'}, inplace=True)
    seat_types = list(seat_data.seating_type)
    seat_type_counts = list(seat_data.counts)

    plt.bar(seat_types, seat_type_counts)
    plt.xticks(rotation=50, fontsize='small', ha='right', y=0.01)
    plt.ylabel('Number of Roller Coasters')
    plt.title('Popularity of Different (non-Sit Down) Seating Types')


def seating_stats(data: pd.DataFrame, stat: str):
    accepted_input = ['speed', 'height', 'length', 'num_inversions']
    if stat not in accepted_input:
        print(f'Sorry, {stat} is not a valid parameter, please try again.')
        return

    filtered = data[data.seating_type != 'na'].dropna().reset_index()
    averages_by_seat = filtered.groupby(filtered.seating_type)[['num_inversions',
                                                                'speed',
                                                                'height',
                                                                'length']].mean().reset_index()
    seat_types = list(averages_by_seat.seating_type)
    stat_by_seat = list(averages_by_seat[stat])
    stat_average = filtered[stat].mean()

    if stat == 'num_inversions':
        stat_name = 'Number of Inversions'
    else:
        stat_name = stat.title()

    ax = plt.subplot()
    plt.bar(seat_types, stat_by_seat)
    plt.xticks(rotation=50, fontsize='small', ha='right', y=0.01)
    plt.ylabel(f'Average {stat_name}', fontsize='small')
    plt.title(f'Average {stat_name} by Seating Type')
    ax.axhline(y=stat_average, label='Global Average', lw=1, color='red')
    plt.legend(fontsize='small')


# seating_popularity(coaster_data)

seating_stats(coaster_data, 'length')
# plt.show()

plt.clf()

# Do roller coaster manufacturers have any specialties (do they focus on speed, height, seating type, or inversions)?



plt.clf()

# Do amusement parks have any specialties?




plt.clf()
