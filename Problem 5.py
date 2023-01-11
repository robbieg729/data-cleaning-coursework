import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Kaggle_survey 2019-2021_cleaned.csv") # read data
df = df[(df["Gender"] == "Male") | (df["Gender"] == "Female")] # filter the data to only include individuals who are Male or Female

year_plot = sns.countplot(data=df, x="Year", hue="Gender") # plot the counts of Males and Females across each year
year_plot.set_title("Number of male and female respondents to 2019-2021 Kaggle surveys") # set title
year_plot.set_xlabel("Year", fontsize=12) # set x-axis label
year_plot.set_ylabel("Count", fontsize=12) # set y-axis label
plt.show()

# Plot the level of education counts, proportionally for males and females. Change some levels first to make them shorter
df["Level of education"].replace(["Some college/university study without earning a bachelor’s degree", 
                                  "No formal education past high school"], ["Some college study", "High school diploma"], inplace=True)
education_plot = sns.histplot(data=df[df["Level of education"] != "I prefer not to answer"], x="Level of education", 
                              hue="Gender", stat="density", common_norm=False, shrink=0.8)
education_plot.set_title("Proportional counts of male and female levels of education")
education_plot.set_xlabel("Level", fontsize=12)
education_plot.set_ylabel("Density", fontsize=12)
plt.show()

# Plot the current role counts, proportionally for males and females
role_plot = sns.histplot(data=df, y="Current role", hue="Gender", stat="density", common_norm=False, shrink=0.8)
role_plot.set_title("Proportional counts of male and female roles")
role_plot.set_xlabel("Density", fontsize=12)
role_plot.set_ylabel("Role", fontsize=12)
plt.show()

# Transform the age column into numerical data, so that we can plot a histogram
df["Age"].replace(["18-21", "22-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-69", "70+"], 
                  [19, 23, 27, 32, 37, 42, 47, 52, 57, 62, 70], inplace=True)
# Plot the distribution of ages, for males and females
age_plot = sns.histplot(data=df, x="Age", hue="Gender", stat="density", common_norm=False, 
                        bins=[15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80])
age_plot.set_title("Distribution of ages for males and females")
age_plot.set_xlabel("Age (years)", fontsize=12)
age_plot.set_ylabel("Density", fontsize=12)
plt.show()

# Transform coding experience column into numerical data to plot a histogram. Note we categorize '0' and '< 1' into the same bin
# of 0-1.
df["Coding experience (years)"].replace(["0", "< 1", 
"1-3", "3-5", "5-10", "10-20", "20+"], [0, 0, 1, 3, 5, 10, 20], inplace=True)
# Plot distribution of coding experience in years, for males and females
experience_plot = sns.histplot(data=df, x="Coding experience (years)", hue="Gender",
                               stat="density", common_norm=False, bins=[0, 1, 3, 5, 10, 20, 30])
experience_plot.set_xticks([0, 1, 3, 5, 10, 20]) # Set x-axis tick labels
experience_plot.set_title("Distribution of coding experience for males and females")    
experience_plot.set_xlabel("Coding experience (years)", fontsize=12)
experience_plot.set_ylabel("Density", fontsize=12)
plt.show()

# Transform yearly compensation column into numerical data, in order to plot a histogram
df["Current yearly compensation ($USD)"].replace(["Not compensated","0-999", "1,000-1,999", "2,000-2,999", 
                    "3,000-3,999", "4,000-4,999", "5,000-7,499", "7,500-9,999", "10,000-14,999", 
                    "15,000-19,999", "20,000-24,999", "25,000-29,999", "30,000-39,999", "40,000-49,999", "50,000-59,999", 
                    "60,000-69,999", "70,000-79,999", "80,000-89,999", "90,000-99,999", "100,000-124,999", "125,000-149,999", 
                    "150,000-199,999", "200,000-249,999", "250,000-299,999", "300,000-499,999", "500,000-999,999", "> 1,000,000"], 
                    [-1, 500, 1500, 2500, 3500, 4500, 6250, 8750, 12500, 17500, 22500, 27500, 35000, 45000, 55000, 65000, 75000, 
                    85000, 95000, 112500, 137500, 175000, 225000, 275000, 400000, 750000, 1000000], inplace=True)
# Plot the distribution of yearly compensation for males and females, on a logarithmic scale
# Note we use a filtered version of the data which does not include any compensation of -1 (initially 'Not compensated') as the 
# majority of these responses were due to students
compensation_plot = sns.histplot(data=df[df["Current yearly compensation ($USD)"] != -1], x="Current yearly compensation ($USD)", 
                                hue="Gender", stat="density", common_norm=False, 
                                bins=np.log10([1, 1000, 2000, 3000, 4000, 5000, 7500, 10000, 20000, 30000, 40000, 50000, 60000, 
                                70000, 80000, 90000, 100000, 150000, 200000, 250000, 300000, 500000, 1000000]), log_scale=True)
compensation_plot.set_title("Distribution of compensation for males and females")
compensation_plot.set_xlabel("Yearly compensation ($USD)", fontsize=12)
compensation_plot.set_ylabel("Density", fontsize=12)
plt.show()

# See list of all countries
countries = list(df["Country"].value_counts(dropna=False).index)
print(countries)

# Sort countries by continent
europe_countries = ["Russia", "United Kingdom", "Germany", "Spain", "France", "Italy", "Turkey", "Poland", "Netherlands",
                    "Ukraine", "Portugal", "Greece", "Sweden", "Switzerland", "Ireland", "Belgium", "Romania", "Belarus",
                    "Czech Republic", "Denmark", "Austria", "Norway", "Hungary"]
north_america_countries = ["United States of America", "Canada", "Mexico"]
south_america_countries = ["Brazil", "Colombia", "Argentina", "Chile", "Peru", "Ecuador"]
asia_countries = ["India", "Japan", "China", "Taiwan", "South Korea", "Pakistan", "Singapore", "Indonesia", "Israel",
                  "Vietnam", "Malaysia", "Thailand", "Bangladesh", "Saudi Arabia", "Iran", "Philippines", "United Arab Emirates",
                  "Hong Kong", "Sri Lanka", "Nepal", "Kazakhstan", "Iraq"]
oceania_countries = ["Australia", "New Zealand"]
africa_countries = ["Nigeria", "Egypt", "South Africa", "Kenya", "Morocco", "Tunisia", "Ghana", "Algeria", "Ethiopia", "Uganda"]

# Create a mapping dictionary for country -> region
region_map = dict()
region_map["Other"] = "Other"
for c in europe_countries:
    region_map[c] = "Europe"
for c in north_america_countries:
    region_map[c] = "North America"
for c in south_america_countries:
    region_map[c] = "South America"
for c in asia_countries:
    region_map[c] = "Asia"
for c in oceania_countries:
    region_map[c] = "Oceania"
for c in africa_countries:
    region_map[c] = "Africa"

# Add a new column to the data called 'Region', which maps all countries to their region (continent)
df.insert(len(df.columns), "Region", df["Country"].map(region_map))
# Plot counts of males and females, proportionally, across all regions (not counting 'Other')
region_plot = sns.histplot(data=df[df["Region"] != "Other"], y="Region", hue="Gender", stat="density", common_norm=False, 
                           shrink=0.8)
region_plot.set_title("Proportion of males and females in each region of the world")
region_plot.set_xlabel("Density", fontsize=12)
region_plot.set_ylabel("Region", fontsize=12)
plt.show()

# Create a facet grid by region, using a filtered version of the data which does not include compensation of -1 or a region of 
# 'Other'
region_facet = sns.FacetGrid(data=df[(df["Region"] != "Other") & (df["Current yearly compensation ($USD)"] != -1)], 
                             col="Region", hue="Gender", col_wrap=3, sharey=False) 
# Plot separate histograms for the yearly compensation for males and females, by region, on a logarithmic scale
region_facet.map_dataframe(sns.histplot, x="Current yearly compensation ($USD)", stat="density", common_norm=False, 
                bins=np.log10([1, 1000, 2000, 3000, 4000, 5000, 7500, 10000, 20000, 30000, 40000, 50000, 60000, 
                        70000, 80000, 90000, 100000, 150000, 200000, 250000, 300000, 500000, 1000000]), log_scale=True)
region_facet.fig.suptitle("Distribution of compensation for males and females by region") # set overall title for facet grid
region_facet.set_xlabels("Current yearly compensation ($USD)", fontsize=12) # Set x-axis labels
region_facet.set_ylabels("Density", fontsize=12) # Set y-axis labels
region_facet.add_legend() # Add legend
plt.show()

# View value counts for all roles
print(df["Current role"].value_counts())
# Get a couple of the top roles, not including 'Student' or 'Other'
top_roles = ["Data Scientist", "Data Analyst"]
# Create new data frame that only includes individuals with one of the top 3 roles above
roles_filtered_df = df[(df["Current role"] == top_roles[0]) | (df["Current role"] == top_roles[1])]
# Create a facet grid by current role, filtering out all individuals with 500 as their yearly compensation
roles_facet = sns.FacetGrid(data=roles_filtered_df[roles_filtered_df["Current yearly compensation ($USD)"] != -1],
                            row="Current role", hue="Gender", sharey=False, aspect=0.5)
# Plot separate histograms for the yearly compensation for males and females, by role, on a logarithmic scale
roles_facet.map_dataframe(sns.histplot, x="Current yearly compensation ($USD)", stat="density", common_norm=False,
                        bins=np.log10([1, 1000, 2000, 3000, 4000, 5000, 7500, 10000, 20000, 30000, 40000, 50000, 60000, 
                        70000, 80000, 90000, 100000, 150000, 200000, 250000, 300000, 500000, 1000000]), log_scale=True)
#roles_facet.fig.suptitle("Distribution of compensation for males and females by role") # set overall title for facet grid
roles_facet.set_xlabels("Current yearly compensation ($USD)", fontsize=12) # Set x-axis labels
roles_facet.set_ylabels("Density", fontsize=12) # Set y-axis labels
roles_facet.add_legend(loc=1, bbox_to_anchor=(0.5, 0.95)) # Add legend
plt.show()