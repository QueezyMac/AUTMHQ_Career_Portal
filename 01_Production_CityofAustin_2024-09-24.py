#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Set up the OpenAI API key
# openai import OpenAI
import openai
from openai import OpenAI
import requests
from bs4 import BeautifulSoup

openai.api_key = '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'
api_key = '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'
API_KEY = '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'
client = OpenAI(api_key='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')

# In[5]:


import feedparser
from urllib.request import urlopen
from bs4 import BeautifulSoup
import os
import csv

# RSS Feed URL
rss_feed = "https://www.austincityjobs.org/postings/search.atom?utf8=%E2%9C%93&query=&query_v0_posted_at_date=day&commit=Search"

# Create a directory to save the job posts if it doesn't exist
output_directory = "Job Posts by URL"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"Folder path created for: {output_directory}")
    
output_directory = "Job Posts by URL"
input_file = os.path.join(output_directory, 'ALL_JOBS.csv')

# Delete the existing ALL_JOBS.csv file if it exists
if os.path.exists(input_file):
    os.remove(input_file)
    print(f"Previous file deleted for: {input_file}")

def extract_data_from_text(content, file_number):
    # Extract the URL from the first line and then remove it
    lines = content.split("\n")
    url = lines[0]
    content = "\n".join(lines[1:])

    markers = {
        "Posting_Title": ("\n Posting Title", "\n Job Requisition Number"),
        "Job_Requisition_Number": ("\n Job Requisition Number", "\n Position Number"),
        "Position_Number": ("\n Position Number", "\n Job Type"),
        "Job_Type": ("\n Job Type", "\n Division Name"),
        "Division_Name": ("\n Division Name", "\n Minimum Qualifications"),
        "Minimum_Qualifications": ("\n Minimum Qualifications", "\n Notes to Applicants"),
        "Preferred_Qualifications": ("\n Preferred Qualifications", "\n Duties, Functions and Responsibilities"),
        "Pay_Range": ("\n Pay Range", "\n Hours "),
        "Hours": ("\n Hours", "\n Job Close Date"),      
        "Job_Close_Date": ("\n Job Close Date", "\n Type of Posting"),
        "Department": ("\n Department", "\n Regular/Temporary"),
        "Category": ("\n Category", "\n Location"),
        "Location": ("\n Location", "\n Preferred Qualifications"),
        "Duties_Functions_and_Responsibilities": ("\n Duties, Functions and Responsibilities ", "\n Knowledge, Skills and Abilities"),
        "Knowledge_Skills_and_Abilities": ("\n Knowledge, Skills and Abilities", "\n Criminal Background Investigation")
    }

    # Extract text based on markers
    extracted_data = {}
    for variable, (start_marker, end_marker) in markers.items():
        start_index = content.find(start_marker) + len(start_marker)
        end_index = content.find(end_marker)
        extracted_data[variable] = content[start_index:end_index].strip()

    # Add URL and file number (formatted with a leading apostrophe) to the extracted data
    extracted_data["URL"] = url
    extracted_data["CSV"] = "'" + file_number  # Prepend an apostrophe to the file number
    
    # Merge fields with single newlines between main sections
    extracted_data["Qualifications"] = f"Minimum Qualification:\n{extracted_data['Minimum_Qualifications']}\n\nPreferred Qualifications:\n{extracted_data['Preferred_Qualifications']}"
    extracted_data["Job_Description"] = f"Duties, Functions and Responsibilities:\n{extracted_data['Duties_Functions_and_Responsibilities']}\n\nKnowledge, Skills and Abilities:\n{extracted_data['Knowledge_Skills_and_Abilities']}"

    return extracted_data

# Fetch and process entries from the RSS Feed
d = feedparser.parse(rss_feed)
for count, entry in enumerate(d['entries']):
    response = urlopen(entry['id'])
    html_content = response.read().decode('utf-8-sig')
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.find_all(string=True)

    blacklist = [
        '[document]',
        'noscript',
        'header',
        'html',
        'meta',
        'head',
        'input',
        'script',
    ]

    output = entry['id'] + "\n"  # Add the URL to the first line
    for t in text:
        if t.parent.name not in blacklist:
            output += '{} '.format(t)

    # Save the extracted content to a .txt file
    txt_filename = os.path.join(output_directory, "{:03}.txt".format(count))
    with open(txt_filename, mode="w", encoding='utf-8-sig') as wfile:
        wfile.write(output)

    # Pass the file number (formatted as a three-digit string) to the function
    file_number = "{:03}".format(count)
    data = extract_data_from_text(output, file_number)
    
    # Save the extracted structured data to a .csv file
    csv_filename = os.path.join(output_directory, f"{file_number}.csv")
    headers_detailed = [
        "URL",
        "CSV",
        "Posting_Title",
        "Job_Requisition_Number",
        "Position_Number",
        "Job_Type",
        "Division_Name",
        "Minimum_Qualifications",
        "Preferred_Qualifications",
        "Qualifications",
        "Pay_Range",
        "Hours",
        "Job_Close_Date",
        "Department",
        "Category",
        "Location",
        "Duties_Functions_and_Responsibilities",
        "Knowledge_Skills_and_Abilities",
        "Job_Description"
    ]
    with open(csv_filename, mode="w", encoding='utf-8-sig', newline='') as wfile:
        writer = csv.DictWriter(wfile, fieldnames=headers_detailed)
        writer.writeheader()
        writer.writerow(data)

# Append extracted data to the ALL_JOBS.csv master file
all_jobs_csv = os.path.join(output_directory, "ALL_JOBS.csv")
write_headers = not os.path.exists(all_jobs_csv)

headers_all_jobs = [
    "URL",
    "CSV",
    "Posting_Title",
    "Job_Requisition_Number",
    "Position_Number",
    "Job_Type",
    "Division_Name",
    "Qualifications",
    "Pay_Range",
    "Hours",
    "Job_Close_Date",
    "Department",
    "Category",
    "Location",
    "Job_Description"
]

with open(all_jobs_csv, 'a', newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=headers_all_jobs)
    if write_headers:
        writer.writeheader()
    
    count = 0
    while True:
        csv_file_path = os.path.join(output_directory, f"{count:03}.csv")
        if not os.path.exists(csv_file_path):
            break
        
        with open(csv_file_path, "r", encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            for row in reader:
                simplified_data = {
                    "URL": row["URL"],
                    "CSV": "'" + "{:03}".format(count),  # Use the count as the three-digit file number for the CSV column and prepend an apostrophe
                    "Posting_Title": row["Posting_Title"],
                    "Job_Requisition_Number": row["Job_Requisition_Number"],
                    "Position_Number": row["Position_Number"],
                    "Job_Type": row["Job_Type"],
                    "Division_Name": row["Division_Name"],
                    "Qualifications": row["Qualifications"],
                    "Pay_Range": row["Pay_Range"],
                    "Hours": row["Hours"],
                    "Job_Close_Date": row["Job_Close_Date"],
                    "Department": row["Department"],
                    "Category": row["Category"],
                    "Location": row["Location"],
                    "Job_Description": row["Job_Description"]
                }
                writer.writerow(simplified_data)
        count += 1

print(f"CSV saved to {all_jobs_csv}")


# In[3]:


#  OPEN AI JOB SUMMARY

import openai
import os
import pandas as pd

def prompt_openai(description):
    """Send a prompt to OpenAI's API and return the response."""
    try:
        description_str = str(description)
        trimmed_description = description_str[:2500]  # Limit the description for prompt size
        
        messages = [
            {"role": "system", "content": "You are a helpful AI summary assistant."},
            {"role": "user", "content": f"Summarize the job description in 20 words:\n\n{trimmed_description}"}
        ]

        # Call the chat API with updated structure if necessary
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Replace with 'gpt-4' if using that model
            messages=messages
        )
        
        # Extract and return the assistant's response
        if response.get('choices') and response['choices'][0].get('message'):
            assistant_message = response['choices'][0]['message']['content']
            return assistant_message.strip()
        else:
            return "No summary available"
        
    except Exception as e:
        print(f"An error occurred in prompt_openai: {e}")
        return "Error generating summary"



# In[6]:


# PYTHON CLEANUP & Addition of markers: '-'

import pandas as pd
import os

# Define input and output directories and files
input_directory = "Job Posts by URL"
output_directory = "Job Posts by URL"
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





# In[7]:


#TEST

import pandas as pd
import os
from datetime import timedelta

# Define input and output directories and files
input_directory = "Job Posts by URL"
output_directory = "Job Posts by URL"
input_file = os.path.join(input_directory, 'ALL_JOBS_SUMMARY_Merged.csv')
output_file = os.path.join(output_directory, 'ALL_JOBS_SUMMARY_Merged_Salary.csv')

# Load the CSV file into a DataFrame
df = pd.read_csv(input_file, encoding='utf-8-sig')

# Check if 'Job_Close_Date' column exists, if not, raise an error
if 'Job_Close_Date' not in df.columns:
    raise ValueError("The 'Job_Close_Date' column is missing from the CSV file.")

# Attempt to convert 'Job_Close_Date' to datetime, coerce errors into NaT
df['Job_Close_Date'] = pd.to_datetime(df['Job_Close_Date'], errors='coerce')

# Handle NaT values by replacing them with a default date or dropping them
# Replace NaT with the current date
default_date = pd.to_datetime('today')
df['Job_Close_Date'].fillna(default_date, inplace=True)

# Calculate 'Job_Open_Date' by subtracting 21 days from 'Job_Close_Date'
df['Job_Open_Date'] = df['Job_Close_Date'] - timedelta(days=21)

# Format 'Job_Close_Date' and 'Job_Open_Date' as date strings in YYYY-MM-DD format
df['Job_Close_Date'] = df['Job_Close_Date'].dt.strftime('%Y-%m-%d')
df['Job_Open_Date'] = df['Job_Open_Date'].dt.strftime('%Y-%m-%d')

# Define a function to calculate the Salary based on the Pay Range
def calculate_salary(pay_range):
    # Check if pay_range is NaN, if so, return "DOE"
    if pd.isna(pay_range):
        return "DOE"
    
    # Extract minimum and maximum values from the pay range
    try:
        min_pay, max_pay = [float(val.replace("$", "").replace("–", "").strip()) for val in pay_range.split() if "$" in val]
        annual_low = min_pay * 2080
        annual_high = max_pay * 2080
        return "${:,.2f} - ${:,.2f} per year".format(annual_low, annual_high)
    except:
        return "DOE"

# Apply the calculate_salary function to the Pay Range column to create the Salary column
df['Salary'] = df['Pay_Range'].apply(calculate_salary)

# Save the updated DataFrame to a CSV file
df.to_csv(output_file, index=False, encoding='utf-8-sig')

# Output the message indicating successful save
print(f"CSV saved to {output_file}")

# Print the 'Job_Close_Date' and 'Job_Open_Date' columns to verify the output
print(df[['Job_Close_Date', 'Job_Open_Date']])


# In[ ]:





# In[8]:


# PYTHON Salary Calculation &  & DATE Calculation 2024.02.13  Addition of markers: '-'


import pandas as pd
import os
from datetime import timedelta

# Define input and output directories and files
input_directory = "Job Posts by URL"
output_directory = "Job Posts by URL"
input_file = os.path.join(input_directory, 'ALL_JOBS_SUMMARY_Merged.csv')
output_file = os.path.join(output_directory, 'ALL_JOBS_SUMMARY_Merged_Salary.csv')

# Load the CSV file into a DataFrame
df = pd.read_csv(input_file, encoding='utf-8-sig')

# Check if 'Job_Close_Date' column exists, if not, raise an error
if 'Job_Close_Date' not in df.columns:
    raise ValueError("The 'Job_Close_Date' column is missing from the CSV file.")

# Attempt to convert 'Job_Close_Date' to datetime, coerce errors into NaT
df['Job_Close_Date'] = pd.to_datetime(df['Job_Close_Date'], errors='coerce')

# Handle NaT values by replacing them with a default date or dropping them
# Replace NaT with the current date
default_date = pd.to_datetime('today')
df['Job_Close_Date'].fillna(default_date, inplace=True)

# Calculate 'Job_Open_Date' by subtracting 21 days from 'Job_Close_Date'
df['Job_Open_Date'] = df['Job_Close_Date'] - timedelta(days=21)

# Define a function to calculate the Salary based on the Pay Range
def calculate_salary(pay_range):
    # Check if pay_range is NaN, if so, return "DOE"
    if pd.isna(pay_range):
        return "DOE"
    
    # Extract minimum and maximum values from the pay range
    try:
        min_pay, max_pay = [float(val.replace("$", "").replace("–", "").strip()) for val in pay_range.split() if "$" in val]
        annual_low = min_pay * 2080
        annual_high = max_pay * 2080
        return "${:,.2f} - ${:,.2f} per year".format(annual_low, annual_high)
    except:
        return "DOE"

# Apply the calculate_salary function to the Pay Range column to create the Salary column
df['Salary'] = df['Pay_Range'].apply(calculate_salary)

# Save the updated DataFrame to a CSV file
df.to_csv(output_file, index=False, encoding='utf-8-sig')

# Output the message indicating successful save and print the 'Job_Close_Date' and 'Job_Open_Date' columns
print(f"CSV saved to {output_file}")
print(df[['Job_Close_Date', 'Job_Open_Date']])


# In[ ]:





# In[9]:


import pandas as pd
import os

# Define input directory and file
input_directory = "Job Posts by URL"  # Make sure this directory exists in your environment
input_file = os.path.join(input_directory, 'ALL_JOBS_SUMMARY_Merged_Salary.csv')  # The CSV file must be in this directory

# Load the CSV file into a DataFrame
df = pd.read_csv(input_file, encoding='utf-8-sig')

# Print column headers vertically
print("Column headers in the file (printed vertically):")
for column in df.columns:
    print(column)


# In[10]:


# MAP THE OUTPUT & PREPARE FOR WIX UPLOAD ! ! !

import pandas as pd
import os

# Define input and output directories and files
input_directory = "Job Posts by URL"
output_directory = "Job Posts by URL"
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
df_output["Job Open Date"] = df_input["Job_Open_Date"]  #Caldulated
df_output["Job Close Date"] = df_input["Job_Close_Date"]
df_output["Company or Organization"] = "City of Austin"
df_output["Business Unit / Division"] = df_input["Department"]
df_output["Job Category"] = df_input["Category"]
df_output["Location"] = "Austin, Texas, United States"
df_output["Qualifications"] = df_input["Qualifications_2"]
df_output["Position Description"] = df_input["Job_Description_2"]
df_output["Expected Salary"] = df_input["Salary"]

# Explicitly convert 'Job_Open_Date' and 'Job_Close_Date' to dates in YYYY-MM-DD format
df_output["Job Close Date"] = pd.to_datetime(df_input["Job_Close_Date"]).dt.date
df_output["Job Open Date"] = pd.to_datetime(df_input["Job_Open_Date"]).dt.date

# Save the output DataFrame with all the headers to the specified output file
df_output.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"CSV saved to {output_file}")


# In[ ]:





# In[ ]:





# In[ ]:




