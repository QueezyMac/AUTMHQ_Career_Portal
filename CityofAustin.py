import openai
from bs4 import BeautifulSoup
import os
import feedparser
from urllib.request import urlopen
from bs4 import BeautifulSoup
import os
import csv
import re
import openai
import os
import pandas as pd

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
        "Pay_Range": ("\n Pay Range", "\n Hours "),
        "Hours": ("\n Hours", "\n Job Close Date"),      
        "Job_Close_Date": ("\n Job Close Date", "\n Type of Posting"),
        "Posting_Type": ("\n Type of Posting", "\n Department "),
        "Department": ("\n Department", "\n Regular/Temporary"),
        "Category": ("\n Category", "\n Location"),
        "Location": ("\n Location", "\n Preferred Qualifications"),
        "Preferred_Qualifications": ("\n Preferred Qualifications", "\n Duties, Functions and Responsibilities"),
        "Duties_Functions_and_Responsibilities": ("\n Duties, Functions and Responsibilities ", "\n Knowledge, Skills and Abilities"),
        "Knowledge_Skills_and_Abilities": ("\n Knowledge, Skills and Abilities", "\n Criminal Background Investigation")
    } 

    # Extract text based on markers
    extracted_data = {}
    end_index = 0
    for variable, (start_marker, end_marker) in markers.items():
        start_index = content.find(start_marker,end_index) + len(start_marker)
        end_index = content.find(end_marker,start_index)
        extracted_data[variable] = content[start_index:end_index].strip()

    # Fix the formatting of the job type
    if(extracted_data["Job_Type"]) == "Full-Time":
        extracted_data["Job_Type"] = "Full time"
    
    elif(extracted_data["Job_Type"] == "Part-Time"):
        extracted_data["Job_Type"] = "Part time"
        
    # Add URL and file number (formatted with a leading apostrophe) to the extracted data
    extracted_data["URL"] = url
    extracted_data["CSV"] = "'" + file_number  # Prepend an apostrophe to the file number
    
    # Merge fields with single newlines between main sections
    extracted_data["Qualifications"] = f"Minimum Qualification:\n{extracted_data['Minimum_Qualifications']}\n\nPreferred Qualifications:\n{extracted_data['Preferred_Qualifications']}"
    extracted_data["Job_Description"] = f"Duties, Functions and Responsibilities:\n{extracted_data['Duties_Functions_and_Responsibilities']}\n\nKnowledge, Skills and Abilities:\n{extracted_data['Knowledge_Skills_and_Abilities']}"

    content = content.upper()
    if("NOTES TO APPLICANTS" in re.sub(r"\s+"," ",extracted_data["Pay_Range"].upper())):
        if("SLARY" in content):
            start_index = content.find("SLARY")
        elif("SALARLY" in content):
            start_index = content.find("SALARLY")
        else:
            start_index = content.find("SALARY")
        end_index = content.find("PAY RANGE")
        salary_text = content[start_index:end_index].strip()
        salary_text = salary_text[:salary_text.find("\n")]
        salary_text = re.sub(r"\s+"," ",salary_text)

        if(salary_text.startswith("SALARY RANGE :")):
            start_index = salary_text.find("SALARY RANGE :")+len("SALARY RANGE :")
            extracted_data["Pay_Range"] = salary_text[start_index:].strip()
        elif(salary_text.startswith("SALARLY RANGES :")):
            start_index = salary_text.find("SALARLY RANGES :")+len("SALARLY RANGES :")
            extracted_data["Pay_Range"] = salary_text[start_index:].strip()
        elif(salary_text.startswith("SALARY RANGE:")):
            start_index = salary_text.find("SALARY RANGE:")+len("SALARY RANGE:")
            extracted_data["Pay_Range"] = salary_text[start_index:].strip()
        elif(salary_text.startswith("SALARY RANGES:")):
            start_index = salary_text.find("SALARY RANGES:")+len("SALARY RANGES:")
            extracted_data["Pay_Range"] = salary_text[start_index:].strip()
        elif(salary_text.startswith("SALARY RANGES :")):
            start_index = salary_text.find("SALARY RANGES :")+len("SALARY RANGES :")
            extracted_data["Pay_Range"] = salary_text[start_index:].strip()
        elif(salary_text.startswith("SLARY RANGES :")):
            start_index = salary_text.find("SLARY RANGES :")+len("SLARY RANGES :")
            extracted_data["Pay_Range"] = salary_text[start_index:].strip()
        elif(salary_text.startswith("SALARY:")):
            start_index = salary_text.find("SALARY:")+len("SALARY:")
            extracted_data["Pay_Range"] = salary_text[start_index:].strip()
    return extracted_data

def prompt_openai(description):
    """Send a prompt to OpenAI's API and return the response."""
    # Convert description to string in case it's not (handles NaN or float)
    description_str = str(description)
    trimmed_description = description_str[:2500]  # Limit the description to 2500 tokens
    messages = [
        {"role": "system", "content": "You are a helpful Ai Summary assistant who returns AI summaries in 1 sentence."},
        {"role": "user", "content": f"Intake the job description, and write a concise and informative 20-word job summary for potential candidates based on the INPUT. End in a period with no additional words.:\n\n{trimmed_description}"}
    ]
    
    response = openai.chat.completions.create(
      model="gpt-4o-mini",
      messages=messages
    )
    
    # Extract the assistant's message from the response
    assistant_message = response.choices[0].message.content
    
    # Return a 20-word summary
    return ' '.join(assistant_message.split()[:80])  # Adjust to 80 based off 3-4 tokens per word


def generate_summaries():
    folder_path = os.path.join(os.getcwd(), "Job Posts by URL")
    csv_path = os.path.join(folder_path, 'ALL_JOBS.csv')
    df = pd.read_csv(csv_path, encoding='utf-8-sig')  # Use the specified encoding

    summary_csv_path = os.path.join(folder_path, 'ALL_JOBS_Summary.csv')

    # Ensure 'Job_AI_Summary' column exists, if not, create it
    if 'Job_AI_Summary' not in df.columns:
        df['Job_AI_Summary'] = ""

    print("Job_AI_Summary\n")

    # Loop through the Job Descriptions and update the Job_AI_Summary column
    # ChatGPT only generates a summary if the job description was updated or a new job was posted
    if(os.path.exists(summary_csv_path)):
        df_summary = pd.read_csv(summary_csv_path)
        for index, row in df.iterrows():
            try:
                df_summary_job_description_no_blanks = re.sub(r"\s+","",df_summary.loc[df_summary['Job_Requisition_Number'] 
                                                                                       == row['Job_Requisition_Number'],
                                                                                       'Job_Description'].item())
                df_job_description_no_blanks = re.sub(r"\s+","",row["Job_Description"])
                if(df_summary_job_description_no_blanks == df_job_description_no_blanks):
                    summary = df_summary.loc[df_summary['Job_Requisition_Number'] == row['Job_Requisition_Number'],
                                'Job_AI_Summary'].item()
                    print(f"CSV {row['CSV']} already generated")
                else:
                    summary = prompt_openai(row['Job_Description'])
            except ValueError:
                summary = prompt_openai(row['Job_Description'])
            df.at[index, 'Job_AI_Summary'] = summary
            print(f"CSV {row['CSV']} : {summary}\n")

        output_path = os.path.join(folder_path, 'ALL_JOBS_SUMMARY.csv')
        df.to_csv(output_path, index=False, encoding='utf-8-sig')  # Use the specified encoding
        print(f"New CSV exported to {output_path}")
    else:
        for index, row in df.iterrows():
            summary = prompt_openai(row['Job_Description'])
            df.at[index, 'Job_AI_Summary'] = summary
            print(f"CSV {row['CSV']} : {summary}\n")

        output_path = os.path.join(folder_path, 'ALL_JOBS_SUMMARY.csv')
        df.to_csv(output_path, index=False, encoding='utf-8-sig')  # Use the specified encoding
        print(f"New CSV exported to {output_path}")

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

# Function to see if a string is a float
def is_float(string):
    try:
        float(string)
        return(True)
    except ValueError:
        return(False)
    
# Function to see if a string is an integer
def is_int(string):
    try:
        int(string)
        return(True)
    except ValueError:
        return(False)

# Define a function to calculate the Salary based on the Pay Range
def calculate_salary(pay_range):
    # Check if pay_range is NaN, if so, return "DOE"
    if pd.isna(pay_range):
        return "DOE","DOE"
    
    # Extract values from the pay range
    try:
        # Input can be in UNPAID, per year, or per hour, so get everything to per hour first or if it is unpaid just return 0
        # for hourly pay rate and salary
        pay_range = pay_range.replace("$ ","$").replace("-", " - ").replace("/HOUR", " per hour").replace("/YEAR", " per year")
        pay_range = re.sub(r" {2,}", " ",pay_range)
        if(pay_range == "UNPAID"):
            return "${:,.2f} per hour".format(0), "${:,.2f} per year".format(0)
        elif(("per year" in pay_range) or ("annually" in pay_range)):
            pay_list = [int(val.replace("$","").replace(",","").strip()) for val in pay_range.split()
                                if (("$" in val) or (is_int(val)))][:2]
            pay_list = [salary/2080 for salary in pay_list]
        else:
            pay_list = [float(val.replace("$", "").strip()) for val in pay_range.split() 
                                if (("$" in val) and (is_float(val.replace("$", "").strip()) 
                                                      and len(val.replace("$", "").strip())==5)) or
                                  ((is_float(val) and len(val)==5))][:2]
        
        # Sometimes the pay is just a single number and not a range, so these if statements allow the pay to be extracted correctly
        if(len(pay_list)==2):
            min_pay,max_pay = pay_list
            annual_low = min_pay * 2080
            annual_high = max_pay * 2080
            return "${:,.2f} - ${:,.2f} per hour".format(min_pay, max_pay), "${:,.2f} - ${:,.2f} per year".format(annual_low, annual_high)
        elif(len(pay_list)==1):
            pay = pay_list[0]
            annual_pay = pay * 2080
            return "${:,.2f} per hour".format(pay), "${:,.2f} per year".format(annual_pay)
        else:
            return "DOE","DOE"
    except Exception as e:
        print(f"Error: {e}")
        return "DOE","DOE"

def main():
    # Set up the OpenAI API key
    api_key = os.getenv("api_key")
    openai.api_key = api_key

    # Create a directory to save the job posts if it doesn't exist
    output_directory = "Job Posts by URL"
    # RSS Feed URL
    rss_feed = "https://www.austincityjobs.org/postings/all_jobs.atom"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Folder path created for: {output_directory}")

    # Delete all the existing files in the "Job Posts by URL" folder besides the ALL_JOBS_SUMMARY.csv
    else:
        for file in os.listdir(output_directory):
            if(file != "ALL_JOBS_SUMMARY.csv"):
                os.remove(os.path.join(output_directory, file))
                print(f"Previous file deleted for: {file}")

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
            "Posting_Type",
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

    #  OPEN AI JOB SUMMARY

    # Call the generate_summaries function
    generate_summaries()

    # PYTHON CLEANUP & Addition of markers: '-'

    # Define input and output directories and files
    input_directory = "Job Posts by URL"
    output_directory = "Job Posts by URL"
    input_file = os.path.join(input_directory, 'ALL_JOBS_SUMMARY.csv')
    output_file = os.path.join(output_directory, 'ALL_JOBS_SUMMARY_Merged.csv')

    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_file, encoding='utf-8-sig')

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
        "Preferred Skills:",
        "Other:"
    ]

    # Apply the format_text function for Qualifications and Job_Description columns
    df['Qualifications_2'] = df['Qualifications'].apply(format_text, headers=qual_headers)
    df['Job_Description_2'] = df['Job_Description'].apply(format_text, headers=job_desc_headers)

    # Save the formatted DataFrame to a CSV file
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"CSV saved to {output_file}")

    # PYTHON Salary Calculation &  & DATE Calculation 2024.02.13  Addition of markers: '-'

    # Define input and output files
    input_file = os.path.join(input_directory, 'ALL_JOBS_SUMMARY_Merged.csv')
    output_file = os.path.join(output_directory, 'ALL_JOBS_SUMMARY_Merged_Salary.csv')

    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_file, encoding='utf-8-sig')

    # Check if 'Job_Close_Date' column exists, if not, raise an error
    if 'Job_Close_Date' not in df.columns:
        raise ValueError("The 'Job_Close_Date' column is missing from the CSV file.")

    # Attempt to convert 'Job_Close_Date' to datetime, coerce errors into NaT
    df['Job_Close_Date'] = pd.to_datetime(df['Job_Close_Date'], errors='coerce')

    # Handle NaT values by replacing them with a blank
    df.fillna({"Job_Close_Date":""}, inplace=True)

    # Job open date is unavailable as it is not on their website
    df['Job_Open_Date'] = ""

    # Format 'Job_Close_Date' and 'Job_Open_Date' as date strings in YYYY-MM-DD format
    df['Job_Close_Date'] = df['Job_Close_Date'].dt.strftime('%Y-%m-%d').fillna("")

    # Apply the calculate_salary function to the Pay Range column to create the Salary column
    for (index,data) in df.iterrows():
        df.at[index,'Pay_Range'], df.at[index,'Salary'] = calculate_salary(data['Pay_Range'])

    # Save the updated DataFrame to a CSV file
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    # Output the message indicating successful save
    print(f"CSV saved to {output_file}")

    # Print the 'Job_Close_Date' and 'Job_Open_Date' columns to verify the output
    print(df[['Job_Close_Date', 'Job_Open_Date']])

    # Define input directory and file
    input_directory = "Job Posts by URL"  # Make sure this directory exists in your environment
    input_file = os.path.join(input_directory, 'ALL_JOBS_SUMMARY_Merged_Salary.csv')  # The CSV file must be in this directory

    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_file, encoding='utf-8-sig')

    # Print column headers vertically
    print("Column headers in the file (printed vertically):")
    for column in df.columns:
        print(column)

    # MAP THE OUTPUT & PREPARE FOR WIX UPLOAD ! ! !

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
        "Job Type (Full, Part, Intern, Co-op)", "AUTMHQ Job Boar... (Job Title, Comp...)", "View Position", "Status", "Sort Order",  
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
    df_output["Job Type (Full, Part, Intern, Co-op)"] = df_input["Job_Type"]
    df_output["Compensation"] = df_input["Pay_Range"]
    df_output["Job Open Date"] = df_input["Job_Open_Date"]  #Caldulated
    df_output["Job Close Date"] = df_input["Job_Close_Date"]
    df_output["Company or Organization"] = "City of Austin"
    df_output["Business Unit / Division"] = df_input["Department"]
    df_output["Job Category"] = "City of Austin"
    df_output["Location"] = "Austin, Texas, United States"
    df_output["Qualifications"] = df_input["Qualifications_2"]
    df_output["Position Description"] = df_input["Job_Description_2"]
    df_output["Expected Salary"] = df_input["Salary"]

    # Explicitly convert 'Job_Close_Date' to dates in YYYY-MM-DD format
    df_output["Job Close Date"] = pd.to_datetime(df_input["Job_Close_Date"]).dt.date

    # Save the output DataFrame with all the headers to the specified output file
    df_output.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"CSV saved to {output_file}")

if __name__ == "__main__":
    main()