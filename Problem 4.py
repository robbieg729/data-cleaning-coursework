import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Kaggle_survey 2019-2021_cleaned.csv") # read data

# filter data by required conditions (Data Scientist and 5+ years of programming)
filtered_df = df[(df["Current role"] == "Data Scientist") &
                 (df["Coding experience (years)"] == "5-10") |
                 (df["Coding experience (years)"] == "10-20") |
                 (df["Coding experience (years)"] == "20+")]

# create 3 new data frames, filtering by year
filtered_2019 = filtered_df[filtered_df["Year"] == 2019]
filtered_2020 = filtered_df[filtered_df["Year"] == 2020]
filtered_2021 = filtered_df[filtered_df["Year"] == 2021]

languages_cols = list()
libraries_cols = list()

# get the relevant columns for the top 5 programming languages and top 5 visualization libraries
for col in filtered_df.columns:
    if "What programming languages do you use on a regular basis?" in col:
        languages_cols.append(col)
    elif "What data visualization libraries or tools do you use on a regular basis?" in col:
        libraries_cols.append(col)

def plot_freq_graph(df, cols, title, xlabel):
    '''
    Plot a frequency graph showing the top 5 of each item from the data
    param df: the data frame
    param cols: the relevant columns to use
    param title: the title for the plot
    param ylabel: the y-axis label for the plot
    '''
    new_df = df[cols] # filter out unwanted columns
    counts = dict()
    for col in new_df.columns:
        item = col[col.index("?") + 1:].strip() # get the name of the item

        # shorten some names
        if item == "Ggplot / ggplot2":
            item = "Ggplot"
        elif item == "Plotly / Plotly Express":
            item = "Plotly"

        # only count options that aren't 'None' and 'Other'    
        if item != "None" and item != "Other":
            counts[item] = new_df[col].value_counts()[1] # add the number of positive responses for that item to the counts dict
    
    # sort the items in descending order of their counts 
    counts = dict(sorted(counts.items(), key=lambda x:x[1], reverse=True)) # https://www.tutorialsteacher.com/articles/sort-dict-by-value-in-python
    
    s = pd.Series(counts) # create a pandas series from the counts
    b = sns.barplot(x=s.index[0:5], y=s[0:5]) # plot the top 5 items from the pandas series of counts
    b.set_xlabel(xlabel=xlabel, fontsize=12) # set x-axis label
    b.set_ylabel("Number of users", fontsize=12) # set y-axis label
    b.set_title(title) # set title
    plt.tight_layout(rect=[0, 0, 1, 1]) # ensure we can see the full plot
    plt.show() # show the plot

# plot top 5 languages for each year
plot_freq_graph(filtered_2019, languages_cols, "Top 5 programming languages used by experienced\ndata scientists in 2019", "Language")
plot_freq_graph(filtered_2020, languages_cols, "Top 5 programming languages used by experienced\ndata scientists in 2020", "Language")
plot_freq_graph(filtered_2021, languages_cols, "Top 5 programming languages used by experienced\ndata scientists in 2021", "Language")

# plot top 5 visualization libraries for each year
plot_freq_graph(filtered_2019, libraries_cols, "Top 5 visualization libraries used by experienced\ndata scientists in 2019", "Library")
plot_freq_graph(filtered_2020, libraries_cols, "Top 5 visualization libraries used by experienced\ndata scientists in 2020", "Library")
plot_freq_graph(filtered_2021, libraries_cols, "Top 5 visualization libraries used by experienced\ndata scientists in 2021", "Library")