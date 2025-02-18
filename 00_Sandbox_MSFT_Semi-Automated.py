#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Set up the OpenAI API key
import openai
import requests
from bs4 import BeautifulSoup
import feedparser
import os


openai.api_key = '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'
api_key = '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'
API_KEY = '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'


# In[ ]:





# In[2]:


# Extract TEXT with Markers

import pandas as pd
import os

# Ensure the "MSFT_JOBS" folder exists relative to the script's location
folder_path = os.path.join(os.getcwd(), "MSFT_JOBS")
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Check for the existence of the Websites.csv file and raise an error if it's not present
csv_path = os.path.join(folder_path, 'Websites.csv')
if not os.path.exists(csv_path):
    raise FileNotFoundError("The file 'Websites.csv' was not found in the 'MSFT_JOBS' folder.")

# Delete all .txt and .csv files in the "MSFT_JOBS" folder, but keep the Websites.csv file
for file in os.listdir(folder_path):
    if (file.endswith('.txt') or file.endswith('.csv')) and file != 'Websites.csv':
        os.remove(os.path.join(folder_path, file))

# Load the Websites.csv file into a DataFrame
df_websites = pd.read_csv(csv_path)

# Generate .txt files from the Websites.csv data
for idx, row in df_websites.iterrows():
    file_name = f"{idx+1:03}.txt"
    with open(os.path.join(folder_path, file_name), 'w', encoding='utf-8-sig') as file:
        file.write(f"URL: {row['URL']}\n")
        file.write(f"URL_Content:\n{row['URL_Content']}\n\n")

# Function to extract data from text, including the city information
def extract_data_from_text(content):
    extracted_data = {}
    markers = {
        "URL": ("URL: ", "\nURL_Content"),
        "Posting_Title": ("URL_Content:\n", "\nSave"),
        "Date_Posted": ("Date posted\n", "\nJob number"),
        "Job_Requisition_Number": ("Job number\n", "\nWork site"),
        "Work_Site": ("Work site\n", "\nTravel"),
        "Travel": ("Travel\n", "\nRole type"),
        "Job_Type": ("Role type\n", "\nProfession"),
        "Profession": ("Profession\n", "\nDiscipline"),
        "Discipline": ("Discipline\n", "\nEmployment type"),
        "Employment_Type": ("Employment type\n", "\nOverview"),
        "Minimum_Qualifications": ("\nQualifications", ["\nPreferred Qualifications", "\nPreferred Qualifications:", "\nAdditional or Preferred Qualifications", "\nAdditional or Preferred Qualifications:"]),
        "Preferred_Qualifications": (["\nPreferred Qualifications", "\nPreferred Qualifications:", "\nAdditional or Preferred Qualifications", "\nAdditional or Preferred Qualifications:"], "The typical base pay range for this role across the U.S. is USD"),
        "Salary_Range": ("The typical base pay range for this role across the U.S. is USD ", "per year."),
        "Duties_Functions_and_Responsibilities": ("Overview\n", "\n\nQualifications"),
        "Knowledge_Skills_and_Abilities": ("Responsibilities\n", "\n\n"),
        # Marker for City extraction
        "City": ("URL_Content:\n", "\n", 3)  # Adjusted to correctly extract the city
    }
    
    for key, marker in markers.items():
        if key == "City":
            # Extracting the city by finding the correct line
            lines = content.split('\n')
            city_line_index = 3  # Adjusted index based on our finding
            extracted_data[key] = lines[city_line_index].strip() if len(lines) > city_line_index else None
        else:
            start_markers = marker[0] if isinstance(marker[0], list) else [marker[0]]
            end_markers = marker[1] if isinstance(marker[1], list) else [marker[1]]

            # Initialize indices
            start_index = -1
            end_index = -1

            # Check each start marker and find the first occurrence
            for start_marker in start_markers:
                temp_start_index = content.find(start_marker)
                if temp_start_index != -1:
                    start_index = temp_start_index + len(start_marker)
                    break

            # If a start marker was found, look for the end marker
            if start_index != -1:
                for end_marker in end_markers:
                    temp_end_index = content.find(end_marker, start_index)
                    if temp_end_index != -1:
                        end_index = temp_end_index
                        break

            # Extract content if valid indices are found
            if start_index != -1 and end_index != -1:
                extracted_text = content[start_index:end_index].strip()
                # Clean up for "Posting_Title" after all extractions
                if key == "Posting_Title":
                    extracted_text = extracted_text.split('\n')[0]  # Take only the first line
                extracted_data[key] = extracted_text
            else:
                extracted_data[key] = None

    return extracted_data


# Process each .txt file and store the results
all_data = []
for idx, row in df_websites.iterrows():
    file_name = f"{idx+1:03}.txt"
    with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8-sig') as file:
        content = file.read()
    extracted_data = extract_data_from_text(content)
    all_data.append(extracted_data)

# Create a DataFrame from the extracted data
all_jobs_df = pd.DataFrame(all_data)

# Create new columns 'Qualifications' and 'Job_Description'
all_jobs_df['Qualifications'] = all_jobs_df['Minimum_Qualifications'].fillna('') + '\n\n' + all_jobs_df['Preferred_Qualifications'].fillna('')
all_jobs_df['Job_Description'] = all_jobs_df['Duties_Functions_and_Responsibilities'].fillna('') + '\n\n' + all_jobs_df['Knowledge_Skills_and_Abilities'].fillna('')

# Clean up the new columns
all_jobs_df['Qualifications'] = all_jobs_df['Qualifications'].str.strip()
all_jobs_df['Job_Description'] = all_jobs_df['Job_Description'].str.strip()

# Save the consolidated DataFrame to a new CSV file
all_jobs_csv_path = os.path.join(folder_path, 'ALL_JOBS.csv')
all_jobs_df.to_csv(all_jobs_csv_path, index=False, encoding='utf-8-sig')

print(f"The consolidated file '{os.path.basename(all_jobs_csv_path)}' has been successfully generated.")


# In[ ]:





# In[3]:


import os
import pandas as pd

# Ensure the "MSFT_JOBS" folder exists relative to the script's location
folder_path = os.path.join(os.getcwd(), "MSFT_JOBS")
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Check for the existence of the ALL_JOBS.csv file and raise an error if it's not present
csv_path = os.path.join(folder_path, 'ALL_JOBS.csv')
if not os.path.exists(csv_path):
    raise FileNotFoundError("The file 'ALL_JOBS.csv' was not found in the 'MSFT_JOBS' folder.")

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_path)

# Get the total number of rows in the DataFrame
num_rows = len(df)

# Create a loop to generate the formatted row numbers and add them to a new column "CSV"
df['CSV'] = ["'{}".format(str(i).zfill(3)) for i in range(1, num_rows + 1)]

# Reorder columns to make 'CSV' the first column
column_order = ['CSV'] + [col for col in df.columns if col != 'CSV']
df = df[column_order]

# Save the updated DataFrame back to the CSV file
df.to_csv(csv_path, index=False, encoding='utf-8-sig')

print(f"Updated CSV file 'ALL_JOBS.csv' with the 'CSV' column as the first column has been successfully saved in the '{folder_path}' folder.")


# In[ ]:





# In[4]:


#  OPEN AI JOB SUMMARY

import openai
import os
import pandas as pd
import string

def clean_text(text):
    """Remove punctuation from the text."""
    return text.translate(str.maketrans('', '', string.punctuation))

def prompt_openai(description):
    """Send a prompt to OpenAI's API and return the response."""
    # Convert description to string in case it's not (handles NaN or float)
    description_str = str(description)
    trimmed_description = description_str[:2500]  # Limit the description to 2500 tokens
    messages = [
        {"role": "system", "content": "You are a helpful Ai Summary assistant who returns AI summaries in 1 sentence."},
        {"role": "user", "content": f"Intake the job description, and write a concise and informative 20-word job summary for potential candidates based on the INPUT. End in a period with no additional words.:\n\n{trimmed_description}"}
    ]
    
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-0613",
      messages=messages
    )
    
    # Extract the assistant's message from the response
    assistant_message = response['choices'][0]['message']['content']
    
    # Return a 20-word summary
    return ' '.join(assistant_message.split()[:80])  # Adjust to 80 based off 3-4 tokens per word


def generate_summaries():
    folder_path = os.path.join(os.getcwd(), "MSFT_JOBS")
    csv_path = os.path.join(folder_path, 'ALL_JOBS.csv')
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    if 'Job_AI_Summary' not in df.columns:
        df['Job_AI_Summary'] = ""

    print("Job_AI_Summary\n")

    for index, row in df.iterrows():
        summary = prompt_openai(row['Job_Description'])
        df.at[index, 'Job_AI_Summary'] = summary
        print(f"CSV {row['CSV']} : {summary}\n")

    output_path = os.path.join(folder_path, 'ALL_JOBS_SUMMARY.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"New CSV exported to {output_path}")

# Call the generate_summaries function
generate_summaries()


# In[ ]:





# In[5]:


# PYTHON CLEANUP & Addition of markers: '-'

import pandas as pd
import os

# Define input and output directories and files
input_directory = "MSFT_JOBS"
output_directory = "MSFT_JOBS"
input_file = os.path.join(input_directory, 'ALL_JOBS_SUMMARY.csv')
output_file = os.path.join(output_directory, 'ALL_JOBS_SUMMARY_Merged.csv')

# Load the CSV file into a DataFrame
df = pd.read_csv(input_file, encoding='utf-8-sig')

# Define a function to format the text based on the provided instructions
def format_text(text, headers):
    if pd.isna(text):  # Check if the text is NaN, if so, return it as is
        return text
    
    # Split the text into lines
    lines = text.split("\n")
    
    # For each line, strip whitespace, then check if it starts with any of the headers.
    # If not, and if the line is not empty, add "- " in front of it
    formatted_lines = [
        line if any(line.strip().startswith(header) for header in headers) or line.strip() == "" else "- " + line for line in lines
    ]
    
    # Join the formatted lines and return
    return "\n".join(formatted_lines)

# Headers for Job_Description and Qualifications
job_desc_headers = [
    "Duties, Functions and Responsibilities:",
    "Responsibilities – Supervisor and/or Leadership Exercised:",
    "Knowledge, Skills and Abilities:",
    "Other:"
    
]

qual_headers = [
    "Minimum Qualification:",
    "Licenses and Certifications Required:",
    "Licenses or Certifications:",
    "Preferred Qualifications:",
    "Preferred Experience",
    "Preferred Skills:",
    "Other:"
]

# Apply the format_text function for Qualifications and Job_Description columns
df['Qualifications_2'] = df['Qualifications'].apply(format_text, headers=qual_headers)
df['Job_Description_2'] = df['Job_Description'].apply(format_text, headers=job_desc_headers)

# Save the formatted DataFrame to a CSV file
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"CSV saved to {output_file}")


# In[ ]:





# In[6]:


# PYTHON Salary & DATE Calculation 2024.02.13 & Addition of markers: '-'

import pandas as pd
from datetime import timedelta
import os

# Define input and output directories and files
input_directory = "MSFT_JOBS"
output_directory = "MSFT_JOBS"
input_file = os.path.join(input_directory, 'ALL_JOBS_SUMMARY_Merged.csv')
output_file = os.path.join(output_directory, 'ALL_JOBS_SUMMARY_Merged_Salary.csv')

# Load the CSV file into a DataFrame
df = pd.read_csv(input_file, encoding='utf-8-sig')

# Ensure "Date_Posted" is in datetime format and calculate "Job Close Date"
if 'Date_Posted' in df.columns:
    df['Date_Posted'] = pd.to_datetime(df['Date_Posted'])
    df['Job_Close_Date'] = df['Date_Posted'] + timedelta(days=21)
else:
    print("Column 'Date_Posted' not found. 'Job Close Date' will not be calculated.")

# Modify Salary_Range to append " per year" to each value
if 'Salary_Range' in df.columns:
    df['Salary_Range'] = df['Salary_Range'].apply(lambda x: f"{x} per year" if pd.notnull(x) else x)

# Define a function to calculate the hourly Pay Range based on the Salary Range
def calculate_hourly_pay(salary_range):
    # Check if salary_range is NaN, if so, return "DOE"
    if pd.isna(salary_range):
        return "DOE"
    # Remove " per year" for calculation
    salary_range = salary_range.replace(" per year", "")
    # Extract minimum and maximum values from the salary range
    try:
        min_salary, max_salary = [float(val.replace("$", "").replace(",", "").strip()) for val in salary_range.split('-')]
        hourly_low = min_salary / 2080
        hourly_high = max_salary / 2080
        return "${:.2f} – ${:.2f} per hour".format(hourly_low, hourly_high)
    except:
        return "DOE"

# Apply the calculate_hourly_pay function to the Salary Range column to create the Pay Range column
df['Pay_Range'] = df['Salary_Range'].apply(calculate_hourly_pay)

# Save the updated DataFrame to a CSV file
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"CSV saved with Pay_Range in $/hour and Job Close Date to {output_file}")

# Print "Salary_Range" and "Job Close Date"
print(df[['Salary_Range', 'Job_Close_Date', 'Date_Posted']])


# In[ ]:





# In[7]:


import pandas as pd
import os

# Define input directory and file
input_directory = "MSFT_JOBS"  # Make sure this directory exists in your environment
input_file = os.path.join(input_directory, 'ALL_JOBS_SUMMARY_Merged_Salary.csv')  # The CSV file must be in this directory

# Load the CSV file into a DataFrame with 'utf-8-sig' encoding
df = pd.read_csv(input_file, encoding='utf-8-sig')

# Print column headers vertically
print("Column headers in the file (printed vertically with 'utf-8-sig' encoding):")
for column in df.columns:
    print(column)


# In[ ]:





# In[8]:


# MAP THE OUTPUT & PREPARE FOR WIX UPLOAD ! ! !

import pandas as pd
import os

# Define input and output directories and files
input_directory = "MSFT_JOBS"
output_directory = "MSFT_JOBS"
input_file = os.path.join(input_directory, 'ALL_JOBS_SUMMARY_Merged_Salary.csv')
output_file = os.path.join(output_directory, 'ALL_JOBS_Ready-2-Upload-2-WIX.csv')

# Load the existing input file into a DataFrame
df_input = pd.read_csv(input_file, encoding='utf-8-sig')

# Create a new DataFrame for the output file with all the specified headers
headers = [
    "Created Date", "Job Title", "Job Requisition Number", "Job_AI_Summary", "Link to Apply", "Compensation", "Expected Salary",
    "Job Open Date", "Job Close Date", "Company or Organization", "Company Logo", "Business Unit / Division", 
    "Job Category", "Qualifications", "Position Description", "Location", 
    "Job Type (Full, Part, Intern)", "AUTMHQ Job Boar... (Job Title, Comp...)", "View Position", "Status", "Sort Order",  
    "ID", "Email Application Materials To:", "Job Level", "AUTMHQ Training Cohort", "Owner", "Updated Date"
]
df_output = pd.DataFrame(columns=headers)

# Initialize the columns of df_output with NaN values
for header in headers:
    df_output[header] = pd.Series([None] * len(df_input))

# Map the input columns to the output columns
df_output["Job Title"] = df_input["Posting_Title"]
df_output["Job_AI_Summary"] = df_input["Job_AI_Summary"]
df_output["Job Requisition Number"] = df_input["Job_Requisition_Number"]
df_output["Job_AI_Summary"] = df_input["Job_AI_Summary"]
df_output["Link to Apply"] = df_input["URL"]
df_output["Job Type (Full, Part, Intern)"] = df_input["Job_Type"]
df_output["Compensation"] = df_input["Pay_Range"]
df_output["Job Open Date"] = df_input["Date_Posted"]
df_output["Job Close Date"] = df_input["Job_Close_Date"]
df_output["Company or Organization"] = "Microsoft"
df_output["Business Unit / Division"] = df_input["Profession"]
df_output["Job Category"] = df_input["Discipline"]
df_output["Location"] = "Austin, TX"
df_output["Qualifications"] = df_input["Qualifications_2"]
df_output["Position Description"] = df_input["Job_Description_2"]
df_output["Location"] = df_input["City"]
df_output["Expected Salary"] = df_input["Salary_Range"]

# Save the output DataFrame with all the headers to the specified output file
df_output.to_csv(output_file, index=False, encoding='utf-8-sig')


print(f"CSV saved to {output_file}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




