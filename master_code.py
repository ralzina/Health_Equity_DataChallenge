#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Read all the information from the APR-DRG dataset
APR_DRG_DATA = pd.read_excel('./datasets/APR-DRG 2020_Export.xlsx')

# Import hospital information
hospitals = pd.read_excel('./datasets/Hospitals_with_Zipcodes.xlsx')

# Rename for merging:
hospitals = hospitals.rename(columns={'Hospital_ID': 'HOSPITAL_ID'})

# Import area type by zip code (rural or urban) from Census zip code data
ruralvsurban = pd.read_csv('./datasets/RuralvsUrban-Data.csv')

# Remove ZCTA5 from Zip Codes
ruralvsurban['NAME'] = ruralvsurban['NAME'].str.replace('ZCTA5 ', '')

# Rename column header as Type for predominantly urban or rural population
ruralvsurban = ruralvsurban.rename(columns={'NAME': 'Zip Code'})
ruralvsurban = ruralvsurban.rename(columns={'Unnamed: 6': 'Type'})

# Import population data
population = pd.read_csv('./datasets/Population-Data.csv')

# Remove ZCTA5 from Zip Codes
population['Zip Code'] = population['Zip Code'].str.replace('ZCTA5 ', '')

# Rename Total to Population
population = population.rename(columns={'Total': 'Population'})

data = APR_DRG_DATA

# Map the zipcodes of each hospital
data = data.merge(hospitals[['HOSPITAL_ID','Zipcode']],on='HOSPITAL_ID',how='left')

# Convert zipcodes to integers
data['Zipcode'] = data['Zipcode'].astype('Int64')

 # Rename Zipcode from severity to Zip Code for merging with median_income
data = data.rename(columns={'Zipcode': 'Zip Code'})

# Make them both be the same datatype in Zip Code for merging
data['Zip Code'] = data['Zip Code'].astype(str)

# Merge rural vs urban based on zip code
data = data.merge(ruralvsurban[['Zip Code', 'Type']], on='Zip Code')

# Make them both be the same datatype in Zip Code for merging
population['Zip Code'] = population['Zip Code'].astype(str)

# Merge with population data
data = data.merge(population[['Zip Code', 'Population']], on='Zip Code')

# Remove non valid values
data = data.dropna(subset=['XC'])
data = data.dropna(subset=['XD'])

# Determine the total population with the same diagnosis classification per zip code

# Group the data by Zip code and add all patients disclosed (PATS)
total_populations = data.groupby('Zip Code').agg({'PATS': 'sum', 'Population': 'first'}).reset_index()

# Determine how much of the population have a diagnosis classification (PATS/Population)
total_populations['PATS/Pop'] = total_populations['PATS'] / total_populations['Population']

# Sort the data to see the ratios
total_populations.sort_values(by=['PATS/Pop'], ascending=False)

from matplotlib.patches import Patch

urban_data = [None] * 5
rural_data = [None] * 5

severity_levels = [0,1,2,3,4]
severity_labels = ['Zero','Minor','Moderate','Major','Extreme']

for i in range(5):
  urban_data[i] = data[(data['Type'] == 'Urban') & (data['SEVERITY'] == severity_levels[i])]['XC'].mean()
  rural_data[i] = data[(data['Type'] == 'Rural') & (data['SEVERITY'] == severity_levels[i])]['XC'].mean()

# X values for the severity levels (0 to 4 for polynomial fitting)
x = np.arange(len(severity_levels))

# Fit the data to linear and quadratic models even though the severity variable is categorical to determine a feel for how each urban and rural are increasing
urban_linear_fit = np.poly1d(np.polyfit(x, urban_data, 1))
urban_quadratic_fit = np.poly1d(np.polyfit(x, urban_data, 2))
rural_linear_fit = np.poly1d(np.polyfit(x, rural_data, 1))
rural_quadratic_fit = np.poly1d(np.polyfit(x, rural_data, 2))

# Calculate R-squared values to get a feel for which line (linear or quadratic) is a better "fit"
urban_linear_r2 = 1 - (np.sum((urban_data - urban_linear_fit(x)) ** 2) / np.sum((urban_data - np.mean(urban_data)) ** 2))
urban_quadratic_r2 = 1 - (np.sum((urban_data - urban_quadratic_fit(x)) ** 2) / np.sum((urban_data - np.mean(urban_data)) ** 2))
rural_linear_r2 = 1 - (np.sum((rural_data - rural_linear_fit(x)) ** 2) / np.sum((rural_data - np.mean(rural_data)) ** 2))
rural_quadratic_r2 = 1 - (np.sum((rural_data - rural_quadratic_fit(x)) ** 2) / np.sum((rural_data - np.mean(rural_data)) ** 2))

# Plot
plt.figure(figsize=(8, 6))

plt.bar(x - 0.2, rural_data, width=0.4, color='blue', label="Rural")
plt.bar(x + 0.2, urban_data, width=0.4, color='gold', label="Urban")

# Generate better quadratic fit
x_fine = np.linspace(0, len(severity_levels) - 1, 100)

# Plot fits
plt.plot(x_fine, rural_quadratic_fit(x_fine), color='blue', linewidth=1.5, label="Rural Quadratic Fit")
plt.plot(x_fine, urban_quadratic_fit(x_fine), color='gold', linewidth=1.5, label="Urban Quadratic Fit")

# Add text box with R^2 values
textstr = '\n'.join((
    f"Urban Quadratic R²: {urban_quadratic_r2:.2f}",
    f"Urban Linear R²: {urban_linear_r2:.2f}",
    f"Rural Quadratic R²: {rural_quadratic_r2:.2f}",
    f"Rural Linear R²: {rural_linear_r2:.2f}"
))
plt.text(0.02, 0.85, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

# Generate legend
legend_elements = [
    Patch(facecolor='blue', edgecolor='blue', label='Rural'),
    Patch(facecolor='gold', edgecolor='gold', label='Urban')
]
plt.legend(handles=legend_elements, loc="upper left")


# Labels and title
plt.xlabel("Severity Level")
plt.ylabel("Average Cost (XC)")
plt.title("Urban vs. Rural XC by Severity")
plt.xticks(x, severity_levels)

plt.show()

# Filter out diagnosis classifications with very few patients to exclude potential outliers (less than 10)
compare = data[data['PATS']>=10]

# Add all the Total Costs and Patients Disclosed for each APRDRG code and find the Total Average Cost by Dividing the sums of TC / PATS

# First, separate into two different dataframes to analyze urban and rural areas separately
urban_data = compare[compare['Type'] == 'Urban']
rural_data = compare[compare['Type'] == 'Rural']

# Then, group by APRDRG code and add all the TC and PATS
grouped_urban_data = urban_data.groupby('APRDRG').agg({'TC': 'sum', 'PATS': 'sum'}).reset_index()
grouped_rural_data = rural_data.groupby('APRDRG').agg({'TC': 'sum', 'PATS': 'sum'}).reset_index()
grouped_urban_data['XC'] = grouped_urban_data['TC'] / grouped_urban_data['PATS']
grouped_rural_data['XC'] = grouped_rural_data['TC'] / grouped_rural_data['PATS']

# Merge the grouped urban and rural dataframes on 'APRDRG' and add suffixes for merging so that each XC is differentiated
compare = pd.merge(grouped_urban_data[['APRDRG', 'XC']], grouped_rural_data[['APRDRG', 'XC']], on='APRDRG', suffixes=('_Urban', '_Rural'))

# Rename the columns
compare = compare.rename(columns={'XC_Urban': 'Urban Cost', 'XC_Rural': 'Rural Cost'})

# Add the foldchange
compare['Foldchange'] = compare['Urban Cost'] / compare['Rural Cost']

# Log2 the foldchange
compare['log2'] = np.log2(compare['Foldchange'])

from scipy import stats

# Perform one-sample t-test to determine if there is a significant difference from zero (no difference between average urban and rural cost for the same diagnosis classifications)
t_statistic, p_value = stats.ttest_1samp(compare['log2'], 0)

# Print the t-test results
print("\nLog 2 fold change:")
print(f" T-statistic: {t_statistic}")
print(f" P-value: {p_value}")

# Calculate the mean of the 'log2' column
mean_log2 = compare['log2'].mean()

# Calculate the median of the 'log2' column
median_log2 = compare['log2'].median()

# Print the mean and median to compare to ensure the distribution is not very skewed
print(f" Mean of log2: {mean_log2}")
print(f" Median of log2: {median_log2}")

print("\n--------------------------------\n")

# Create the box plot with compare
plt.figure(figsize=(8, 6))
sns.boxplot(y="log2", data=compare)
plt.title("Box Plot of log2 Fold Change (Urban Cost / Rural Cost)")
plt.ylabel("log2 Fold Change")

# Add a horizontal line at y=0
plt.axhline(y=0, color='red', linestyle='--')
plt.show()
print(compare['log2'].describe())
print()

# Calculate total rural and urban patients
total_rural_patients = data[data['Type'] == 'Rural']['PATS'].sum()
total_urban_patients = data[data['Type'] == 'Urban']['PATS'].sum()

# Calculate total number of hospitals
total_hospitals = data['HOSPITAL_ID'].nunique()

# Calculate number of urban and rural zip codes
urban_zip_codes = data[data['Type'] == 'Urban']['Zip Code'].nunique()
rural_zip_codes = data[data['Type'] == 'Rural']['Zip Code'].nunique()


print("\n--------------------------------\n")

# Print the results
print("Rural vs. Urban general comparison:")
print(f" Total Rural Patients: {total_rural_patients}")
print(f" Total Urban Patients: {total_urban_patients}")
print(f" Total Hospitals: {total_hospitals}")
print(f" Urban Zip Codes: {urban_zip_codes}")
print(f" Rural Zip Codes: {rural_zip_codes}")



zip_code_data = data.groupby(['Zip Code', 'Type', 'Population'])['PATS'].sum().reset_index()

# Display Population and Patients disclosed for each urban and rural

# Group by 'Type' and calculate the sum of 'Population' and 'PATS'
type_data = zip_code_data.groupby('Type').agg({'Population': 'sum', 'PATS': 'sum'})

# Reset the index to make 'Type' a column
type_data = type_data.reset_index()

# Display the DataFrame

"""# Average Cost Distribution Rural Vs Urban"""

# Create violin plot for average cost for each urban and rural
plt.figure(figsize=(12, 6))
sns.violinplot(x='Type', y='XC', data=data)

outliers = len(data[(data['Type'] == 'Urban') & (data['XC'] > 4e5)])

plt.title('Average Cost by Urban and Rural Hospitals')
plt.xlabel('Location')
plt.ylabel('Average Cost in Millions')
plt.show()

# Create violin plot for average cost for each urban and rural cut off at 400,000

plt.figure(figsize=(12, 6))

# only include data less than 400,000
sns.violinplot(x='Type', y='XC', data=data[data['XC'] < 4e5])

plt.title('Average Cost by Urban and Rural Hospitals (XC < 4e5)')
plt.xlabel('Location')
plt.ylabel('Average Cost')
plt.show()

# Do a significance test for the non-normal distribution to determine if these different distributions could have occurred randomly
from scipy import stats

# Separate data for urban and rural areas
urban_cost = data[data['Type'] == 'Urban']['XC']
rural_cost = data[data['Type'] == 'Rural']['XC']

"""# Explanation of Violin plot
**Black Box**
* The black box refers to the 50% of the data, meaning that 50% of the values are inside the thick black box.

**Vertical black line**
* The vertical black line is the range of everything that can be considered a value within the range. Anything that's outside of this line is considered an outlier.

**Horizontal White Line**
* The horizontal white line is the median value of the data.

# Average Cost Differences Discussed
1. Urban costs are much higher than rural, but also have much more variability than rural costs.
2. Urban zip codes could have better quality of treatments do to the higher costs, but the high varibility means that there is a wide difference in what people pay. Prices may not be fair in certain cases.
3. Rural zip codes may have less variability and less costs, but this may be misleading as there could be less quality treatments and less access to advanced equipment.
"""
