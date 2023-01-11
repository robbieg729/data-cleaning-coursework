import pandas as pd
import numpy as np

df = pd.read_csv("Kaggle_survey 2019-2021.csv") # Read in data frame
df.drop(columns="Unnamed: 0", inplace=True) # Drop first column since it is not needed
df = df[df["Duration (in seconds)"] >= 120] # Filter out all participants who took less than 2 minutes

# Replace some column names to make them a bit tidier
df.rename(columns={"What is your gender? - Selected Choice": "Gender", "What is your age (# years)?": "Age", 
                   "In which country do you currently reside?": "Country", 
                   "What is the highest level of formal education that you have attained or plan to attain within the next 2 years?": "Level of education",
                   "Select the title most similar to your current role (or most recent title if retired): - Selected Choice": "Current role",
                   "What is your current yearly compensation (approximate $USD)?": "Current yearly compensation ($USD)",
                   "For how many years have you been writing code and/or programming?": "Coding experience (years)",
                   "What is the primary tool that you use at work or school to analyze data? (Include text response) - Selected Choice": "Primary tool to analyze data",
                   "What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice": "What programming language would you recommend an aspiring data scientist to learn first?",
                   "Approximately how much money have you (or your team) spent on machine learning and/or cloud computing services at home (or at work) in the past 5 years (approximate $USD)?": "Money spent on machine learning and/or cloud computing services in the past 5 years ($USD)",
                   "What is the size of the company where you are employed?": "Number of employees at your company"}, inplace=True)

print(df["Age"].value_counts(dropna=False)) # All fine, no intersecting age ranges and no NaN

# No NaN values, but some repeated levels like 'Other' and 'I do not wish to disclose my location'. 'South Korea' is also
# repeated as 'Republic of Korea'. Also a few country names can be tidied up.   
print(list(df["Country"].value_counts(dropna=False).index))
df["Country"].replace(["I do not wish to disclose my location", "Iran, Islamic Republic of...", "Hong Kong (S.A.R.)", 
                       "United Kingdom of Great Britain and Northern Ireland", "Viet Nam", "Republic of Korea"], 
                      ["Other", "Iran", "Hong Kong", "United Kingdom", "Vietnam", "South Korea"], inplace=True) 

# 471 NaN values and some overlapping levels, such as 'Currently not employed' and 'Not employed'. Change the overlapping
# levels to be the same, and change NaN to 'Not employed'. Also unclear if some levels are the same (or 
# if they have been worded differently across years), e.g. 'Data Scientist' and 'Data Analyst'. By observing 
# the counts for just 2019, we see the survey has defined them to be different.
print(df["Current role"].value_counts(dropna=False))
print(df[df["Year"] == 2019]["Current role"].value_counts(dropna=False))
df["Current role"].replace([np.nan, "Currently not employed", "Program/Project Manager", "Product Manager"], ["Not employed", "Not employed", "Product/Project Manager", "Product/Project Manager"], inplace=True)

print(df["Level of education"].value_counts(dropna=False)) # No repeats, but 260 NaN
df["Level of education"].fillna("I prefer not to answer", inplace=True) # Replace NaN with existing category of 'I prefer not to answer'

# No intersecting ranges here, but NaN is the most common with over 20,000 values. In line 42, we observe that most of this is
# because of students who do not have a place of business. Hence we can replace NaN values with "I do not know / not employed"
print(df["Approximately how many individuals are responsible for data science workloads at your place of business?"].value_counts(dropna=False))
print(df[df["Current role"] == "Student"]["Approximately how many individuals are responsible for data science workloads at your place of business?"].value_counts(dropna=False))
df["Approximately how many individuals are responsible for data science workloads at your place of business?"].fillna("I do not know / not employed", inplace=True)

# No NaN values, but Man, Male, Woman, Female are all separated, so need to replace values so that they are the same category
print(df["Gender"].value_counts(dropna=False)) 
df["Gender"].replace(["Man", "Woman"], ["Male", "Female"], inplace=True)

# Similar situation where NaN is by far the most common and is present in around 1/3 of the responses, but again. this is mostly
# due to students who are not employed. Hence replace NaN with "I do not know / not employed". Also have repeated levels of "> 10,000 employees"
# and "10,000 ore more employees". We will also get rid of the word "employees" from all levels since column name was changed
print(df["Number of employees at your company"].value_counts(dropna=False))
print(df[df["Current role"] == "Student"]["Number of employees at your company"].value_counts(dropna=False))
df["Number of employees at your company"].replace([np.nan, "0-49 employees", "1000-9,999 employees", "50-249 employees",
                                                   "10,000 or more employees", "> 10,000 employees", "250-999 employees"], 
                                                   ["I do not know / not employed", "0-49", "1,000-9,999", "50-249", "> 10,000", 
                                                    "> 10,000", "250-999"], inplace=True)

# Again have NaN in around 1/3 of the responses, again mostly due to students. Replace all NaN with "I do not know / not employed"
print(df["Does your current employer incorporate machine learning methods into their business?"].value_counts(dropna=False))
print(df[df["Current role"] == "Student"]["Does your current employer incorporate machine learning methods into their business?"].value_counts(dropna=False))
df["Does your current employer incorporate machine learning methods into their business?"].replace([np.nan, "I do not know"], 
                                                                                            "I do not know / not employed", inplace=True)

# NaN accounting for around 1/3 of responses, again largely due to students who don't get compensated. Can replace all NaN with
# new level 'Not compensated'. Even though this would be the same as being in the level of "$0-999", it means we can separate
# those who are not compensated and those who are compensated but receive less than $1,000. Also remove $ sign from all levels,
# and we have some repeats.
print(df["Current yearly compensation ($USD)"].value_counts(dropna=False))
print(df[df["Current role"] == "Student"]["Current yearly compensation ($USD)"].value_counts(dropna=False))
df["Current yearly compensation ($USD)"].fillna("Not compensated", inplace=True)
df["Current yearly compensation ($USD)"].replace(["$0-999", "300,000-500,000", "> $500,000", "$500,000-999,999", ">$1,000,000"], ["0-999", "300,000-499,999", "500,000-999,999", "500,000-999,999", "> 1,000,000"], inplace=True)

# NaN accounting for 1/3 of responses again due to students. Just replace with 0 this time since there is no range,
# and also get rid of repeat levels and $ sign from all levels.
print(df["Money spent on machine learning and/or cloud computing services in the past 5 years ($USD)"].value_counts(dropna=False))
print(df[df["Current role"] == "Student"]["Money spent on machine learning and/or cloud computing services in the past 5 years ($USD)"].value_counts(dropna=False))
df["Money spent on machine learning and/or cloud computing services in the past 5 years ($USD)"].fillna("0", inplace=True)
df["Money spent on machine learning and/or cloud computing services in the past 5 years ($USD)"].replace(["$0 (USD)", "$0 ($USD)", "$100,000 or more ($USD)", "> $100,000 ($USD)", "$100-$999", "$1000-$9,999", "$1-$99", "$10,000-$99,999"], ["0", "0", "> 100,000", "> 100,000", "100-999", "1,000-9,999", "1-99", "10,000-99,999"], inplace=True)

# Around 16,000 NaN responses. Replacing them with existing 'Other' category doesn't make the most sense since
# a NaN response is probably more likely to correspond to an individual not actually using a tool to analyze data, since they
# may not be a data scientist/analyst. Hence replace with 'Do not analyze data'
print(df["Primary tool to analyze data"].value_counts(dropna=False))
df["Primary tool to analyze data"].fillna("Do not analyze data", inplace=True)

# Less than 4,000 NaN responses, replace them with 0 category. Get rid of the word 'years' in all
# levels to make the data a bit more numeric, and replace 'I have never written code' with 0. Also '1-2 years' and '1-3 years'
# are both different levels. As we also have '3-5 years', we can replace all '1-2 years' with '1-3'.  
print(df["Coding experience (years)"].value_counts(dropna=False))
df["Coding experience (years)"].replace([np.nan, "I have never written code", "< 1 years", "1-2 years", "1-3 years", "3-5 years", 
                                        "5-10 years", "10-20 years", "20+ years"], 
                                        ["0", "0", "< 1", "1-3", "1-3","3-5", "5-10", "10-20", "20+"], inplace=True)

# Less than 7,000 NaN responses, put into existing 'None' category
print(df["What programming language would you recommend an aspiring data scientist to learn first?"].value_counts(dropna=False))
df["What programming language would you recommend an aspiring data scientist to learn first?"].fillna("None", inplace=True)

# Around 8,000 NaN responses, put into existing 'Never' category which already accounts for around 40,000 responses.
# Also some overlap in response categories across years, so need to replace some of them
print(df["Approximately how many times have you used a TPU (tensor processing unit)?"].value_counts(dropna=False))
df["Approximately how many times have you used a TPU (tensor processing unit)?"].fillna("Never", inplace=True)
df["Approximately how many times have you used a TPU (tensor processing unit)?"].replace(["6-25 times", "More than 25 times"], ["6-24 times", "> 25 times"], inplace=True)

# Under 10,000 NaN responses, most likely would be existing 'I do not use machine learning methods' category.
# Also some overlap in choices across years, so need to replace some categories
print(df["For how many years have you used machine learning methods?"].value_counts(dropna=False))
df["For how many years have you used machine learning methods?"].fillna("I do not use machine learning methods", inplace=True)
df["For how many years have you used machine learning methods?"].replace(["Under 1 year", "< 1 years", "10-15 years", "20 or more years"], ["< 1 year", "< 1 year", "10-20 years", "20+ years"], inplace=True)

# Get all columns corresponding to multiple selection questions
multiple_selection_cols = list() # list of all multiple selection columns
new_col_names = dict() # dictionary to map all multiple selection column names to other names that are cleaner
for col in df.columns:
    if "Selected Choice -" in col:
        multiple_selection_cols.append(col)
        option = col[col.index("Choice -") + 9:].strip() # get the option
        new_col_names[col] = col[0:col.index("(Select all that apply)") - 1] + " " + option # add the new column name to the dictionary

# Replaces all NaN values with 0 in all multiple selection columns, and all non-NaN values with 1. Essentially corresponds to
# a 1 if the respondent did select that choice, and a 0 if they did not
df[multiple_selection_cols] = df[multiple_selection_cols].notnull().astype("int")

df.rename(columns=new_col_names, inplace=True) # rename the multiple selection columns

df.to_csv("Kaggle_survey 2019-2021_cleaned.csv") # save cleaned data frame to new CSV file