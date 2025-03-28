import os
import openai
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import StaleElementReferenceException
import time
import pandas as pd
import re

# Function to get the full html code and urls for all the jobs 
def get_job_sources(url):
     driver = webdriver.Chrome()
     current_page_url = url
     # Open page
     driver.get(url)
     # Microsoft job board uses cookies, so ensure the parts that use cookies are in the extracted code
     cookies_dict = driver.get_cookies()
     for cookie_dict in cookies_dict:
          driver.add_cookie(cookie_dict)
     # Sleep for 10 seconds to ensure page has fully loaded
     time.sleep(10)
     # Find all the buttons to the full job details on the page
     jobs = driver.find_elements(By.XPATH, "//button[text() = 'See details']")
     next_page = driver.find_elements(By.XPATH, "//button[@aria-label = 'Go to next page' "
                                      "and not(@aria-disabled = 'true')]")
     page_sources = []
     urls = []
     i=0
     while(i<len(jobs) or next_page):
          try:
               if(i==len(jobs) and next_page):
                    next_page = driver.find_elements(By.XPATH, "//button[@aria-label = 'Go to next page']")
                    next_page[0].click()
                    time.sleep(5)
                    current_page_url = driver.current_url
                    jobs = driver.find_elements(By.XPATH, "//button[text() = 'See details']")
                    i=0
                    next_page = driver.find_elements(By.XPATH, "//button[@aria-label = 'Go to next page' "
                                                     "and not(@aria-disabled = 'true')]")

               else:
                    jobs[i].click()
                    time.sleep(5)
                    page_sources.append(driver.page_source)
                    urls.append(driver.current_url)
                    driver.get(current_page_url)
                    time.sleep(5)
                    i+=1
          except StaleElementReferenceException:
               print("Error: Stale element reference exception")
               jobs = driver.find_elements(By.XPATH, "//button[text() = 'See details']")
     driver.quit()
     return page_sources, urls

# Function to scrape a page on the job board to extract job information and store information in a CSV file
def get_job_info(html_source, url, file_number, directory="MSFT_JOBS"):
     job_info = {}
     soup = BeautifulSoup(html_source, "html.parser")
     job_info["URL"] = url
     job_info["CSV"] = "'{:03}".format(file_number)
     # Get job title
     job_info["Job Title"] = soup.select("div.ms-DocumentCard.SearchJobDetailsCard h1")[0]
     job_info["Job Title"] = job_info["Job Title"].string
     # Get job location
     job_info["Location"] = soup.select("div.ms-DocumentCard.SearchJobDetailsCard div.ms-Stack-inner")[0]
     job_info["Location"] = job_info["Location"].get_text()
     # Extract top section of job information
     top_section = soup.select("div.ms-DocumentCard.SearchJobDetailsCard div.IyCDaH20Khhx15uuQqgx")
     top_section = [section.get_text("\n") for section in top_section]
     for data in top_section:
          [key, value] = data.split("\n",1)
          job_info[key] = value
     # Extract main body of job information
     main_body_html = soup.select("div.ms-DocumentCard.SearchJobDetailsCard div.fcUffXZZoGt8CJQd8GUl>div")
     # Ensure text in tags are formatted correctly and store each section of the main body in a main body list
     # To ensure text is formatted correctly remove the span, strong, a, b, i, and sup tags before storing in the list
     main_body = []
     for tags in main_body_html:
          remove_tags = tags.find_all("span")
          for span_tag in remove_tags:
               span_tag.unwrap()

          remove_tags = tags.find_all("strong")
          for strong_tag in remove_tags:
               strong_tag.unwrap()

          remove_tags = tags.find_all("a")
          for a_tag in remove_tags:
               a_tag.unwrap()

          remove_tags = tags.find_all("b")
          for b_tag in remove_tags:
               b_tag.unwrap()

          remove_tags = tags.find_all("i")
          for i_tag in remove_tags:
               i_tag.unwrap()
          
          remove_tags = tags.find_all("sup")
          for sup_tag in remove_tags:
               sup_tag.unwrap()

          tags = BeautifulSoup(str(tags),"html.parser")
          main_body.append(tags.get_text("\n"))

     # Main body contains Overview, Responsibilities, and Qualifications
     # Typically, the main body is split into 3 div tags which is a child of a div tag with class fcUffXZZoGt8CJQd8GUl.
     # However, sometimes the main body only contains 2 div tags which are children of a div tag with class fcUffXZZoGt8CJQd8GUl
     # and one of the div tags are empty. The not empty div tag contains everything the main body should contain.
     for item in main_body:
          if(item != "" and (item.startswith("Overview") or item.startswith("Responsibilities") or item.startswith("Qualifications"))):
               [key, value] = item.split("\n",1)
               job_info[key] = value
          
          elif(item.__contains__("Overview") and item.__contains__("Responsibilities") and item.__contains__("Qualifications")):
               job_info["Overview"] = item[item.find("Overview")+len("Overview"):item.find("Responsibilities")].replace(":","")
               job_info["Responsibilities"] = item[item.find("Responsibilities")+len("Responsibilities"):item.find("Qualifications")].replace(":","")
               job_info["Qualifications"] = item[item.find("Qualifications")+len("Qualifications"):].replace(":","")
               break

          elif(not(item.__contains__("Overview")) and item.__contains__("Responsibilities") and item.__contains__("Qualifications")):
               job_info["Overview"] = item[item.find("Job Description")+len("Job Description"):item.find("Responsibilities")].replace(":","")
               job_info["Responsibilities"] = item[item.find("Responsibilities")+len("Responsibilities"):item.find("Qualifications")].replace(":","")
               job_info["Qualifications"] = item[item.find("Qualifications")+len("Qualifications"):].replace(":","")
               break

          elif(not(item.__contains__("Overview")) and item.__contains__("You might thrive in this role if:") and item.__contains__("Qualifications")):
               job_info["Overview"] = item[item.find("Job Description")+len("Job Description"):item.find("Responsibilities")].replace(":","")
               job_info["Responsibilities"] = item[item.find("You might thrive in this role if:")+len("You might thrive in this role if:"):item.find("Qualifications")].replace(":","")
               job_info["Qualifications"] = item[item.find("Qualifications")+len("Qualifications"):].replace(":","")
               break

          elif(item.__contains__("Overview") and item.__contains__("You might thrive in this role if:") and item.__contains__("Qualifications")):
               job_info["Overview"] = item[item.find("Overview")+len("Overview"):item.find("You might thrive in this role if:")].replace(":","")
               job_info["Responsibilities"] = item[item.find("You might thrive in this role if:")+len("You might thrive in this role if:"):item.find("Qualifications")].replace(":","")
               job_info["Qualifications"] = item[item.find("Qualifications")+len("Qualifications"):].replace(":","")
               break

          else:
               job_info["Overview"] = "Error getting job overview"
               job_info["Responsibilities"] = "Error getting job responsibilities"
               job_info["Qualifications"] = "Error getting qualifications for the job"
               break
     # Store Job information in a CSV file
     job_info_df = pd.DataFrame([job_info])
     path = os.path.join(directory, "{:03}.csv".format(file_number))
     job_info_df.to_csv(path, index=False)

# Function to get the job info for all the jobs from the html code and urls for each job page
# Stores each job in a seperate csv file
def get_all_jobs_info(page_sources, urls, directory="MSFT_JOBS"):
     i=0
     while(i<len(page_sources) and i<len(urls)):
          get_job_info(page_sources[i], urls[i], i, directory)
          i+=1


# Function to merge all of the CSV files to one file
def get_all_jobs_info_csv(directory="MSFT_JOBS"):
     count = 0
     df = pd.DataFrame()
     file = os.path.join(directory, "{:03}.csv".format(count))
     while(os.path.exists(file)):
          new_row = pd.read_csv(file)
          df = pd.concat([df, new_row], ignore_index=True)
          count += 1
          file = os.path.join(directory, "{:03}.csv".format(count))
     
     df["Job_Description"] = ""
     df["Salary"] = ""
     df["Closing Date"] = ""
     for (index, data) in df.iterrows():
          # Job description is the overview and the responsibilities
          if (not pd.isna(data["Responsibilities"]) and not pd.isna(data["Overview"])):
               df.at[index,"Job_Description"] = ("Overview:"+"\n"+data["Overview"]
               +"\n\n\nResponsibilities:"+"\n"+data["Responsibilities"])

          elif(not pd.isna(data["Responsibilities"]) and pd.isna(data["Overview"])):
               df.at[index,"Job_Description"] = ("Overview:"+"\n"+"Error getting job overview"
               +"\n\n\nResponsibilities:"+"\n"+data["Responsibilities"])

          elif(pd.isna(data["Responsibilities"]) and not pd.isna(data["Overview"])):
               df.at[index,"Job_Description"] = ("Overview:"+"\n"+data["Overview"]
               +"\n\n\nResponsibilities:"+"\n"+"Error getting job responsibilities")

          else:
               df.at[index,"Job_Description"] = ("Overview:"+"\n"+"Error getting job overview"
               +"\n\n\nResponsibilities:"+"\n"+"Error getting job responsibilities")

          # Extract salary range for job if it is in the job description
          # Salary is stored in the qualifications section
          lines = data["Qualifications"].split("\n")
          count = 0
          for line in lines:
               if("base pay range" in line and df.at[index,"Salary"]==""):
                    df.at[index,"Salary"] = line[line.find("USD")+len("USD "):line.find("USD")+line[line.find("USD"):].find(".")]
                    df.at[index, "Qualifications"] = "\n".join(lines[:count])

               elif("Microsoft will accept applications for the role until " in line and df.at[index,"Closing Date"]==""):
                    df.at[index,"Closing Date"] = line[line.find("Microsoft will accept applications for the role until ")+
                                                       len("Microsoft will accept applications for the role until "):
                                                       line.find("Microsoft will accept applications for the role until ")
                                                       +line[line.find("Microsoft will accept applications for the role until "):].find(".")]
               count += 1

     column_list = ["URL","CSV","Job Title","Location","Date posted","Closing Date","Job number","Work site","Travel","Role type",
              "Profession","Discipline","Employment type","Overview","Qualifications","Responsibilities","Job_Description",
              "Salary"]
     for column in column_list:
          if(not(column in df.columns)):
               df[column] = ""
     df = df[column_list]
     df.to_csv(os.path.join(directory, "ALL_JOBS.csv"), index=False)


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
    folder_path = os.path.join(os.getcwd(), "MSFT_JOBS")
    csv_path = os.path.join(folder_path, 'ALL_JOBS.csv')
    df = pd.read_csv(csv_path, encoding='utf-8-sig')  # Use the specified encoding

    summary_csv_path = os.path.join(folder_path, 'ALL_JOBS_Summary.csv')

    # Ensure 'Job_AI_Summary' column exists, if not, create it
    if 'Job_AI_Summary' not in df.columns:
        df['Job_AI_Summary'] = ""

    print("Job_AI_Summary\n")

    # Loop through the Job Descriptions and update the Job_AI_Summary column
    if(os.path.exists(summary_csv_path)):
        df_summary = pd.read_csv(summary_csv_path)
        for index, row in df.iterrows():
            # ChatGPT only generates a summary if the job description was updated or a new job was posted
            if(not df_summary.loc[df_summary['Job number'] == row['Job number'],
                              'Job_Description'].empty):
                df_summary_job_description_no_blanks = re.sub(r"\s+","",df_summary.loc[df_summary['Job number'] 
                                                                                       == row['Job number'],
                                                                                       'Job_Description'].item())
                df_job_description_no_blanks = re.sub(r"\s+","",row["Job_Description"])
                if(df_summary_job_description_no_blanks == df_job_description_no_blanks):
                    summary = df_summary.loc[df_summary['Job number'] == row['Job number'],
                                'Job_AI_Summary'].item()
                    print(f"CSV {row['CSV']} already generated")
                else:
                    summary = prompt_openai(row['Job_Description'])
            else:
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
        line if any(line.upper().strip().startswith(header.upper()) for header in headers) or line.strip() == "" else "- " + line for line in lines
    ]
    
    # Join the formatted lines and return
    return "\n".join(formatted_lines)

# Define a function to calculate the hourly Pay Range based on the Salary Range
def calculate_hourly_pay(salary_range):
    per_hour = False
    per_month = False
    per_year = False
    # Check if salary_range is NaN, if so, return "DOE"
    if pd.isna(salary_range):
        return "DOE","DOE"
    # Remove " per year" or " per month" for calculation
    elif(salary_range[-10:] == " per month"):
        salary_range = salary_range.replace(" per month", "")
        per_month = True
    elif(salary_range[-9:] == " per hour"):
        salary_range = salary_range.replace(" per hour", "")
        per_hour = True
    else:
        salary_range = salary_range.replace(" per year", "")
        per_year = True
    # Extract minimum and maximum values from the salary range
    try:
        min_salary, max_salary = [float(val.replace("$", "").replace(",", "").strip()) for val in salary_range.split('-')]
        if(per_year):
            hourly_low = min_salary / 2080
            hourly_high = max_salary / 2080
        elif(per_month):
            hourly_low = min_salary / 160
            hourly_high = max_salary / 160
            min_salary = min_salary*12
            max_salary = max_salary*12
        elif(per_hour):
             hourly_low = min_salary
             hourly_high = max_salary
             min_salary = min_salary*40*52
             max_salary = max_salary*40*52
        return "${:.2f} – ${:.2f} per year".format(min_salary, max_salary), "${:.2f} – ${:.2f} per hour".format(hourly_low, hourly_high)
    except:
        return "DOE", "DOE"
    
def main():
     # Set up the OpenAI API key
    api_key = os.getenv("api_key")
    openai.api_key = api_key

     # Clear files in MSFT_JOBS folder besides ALL_JOBS_SUMMARY.csv
    for file in os.listdir("MSFT_JOBS"):
        if(file != "ALL_JOBS_SUMMARY.csv"):
            os.remove(os.path.join("MSFT_JOBS", file))

     # Extract job information
    url = "https://jobs.careers.microsoft.com/global/en/search?lc=Austin%2C%20Texas%2C%20United%20States&l=en_us&pg=1&pgSz=20&o=Relevance&flt=true"
    [page_sources, urls] = get_job_sources(url)
    get_all_jobs_info(page_sources, urls)
    get_all_jobs_info_csv()

    # Call the generate_summaries function
    generate_summaries()

    # Define input and output directories and files
    input_directory = "MSFT_JOBS"
    output_directory = "MSFT_JOBS"
    input_file = os.path.join(input_directory, 'ALL_JOBS_SUMMARY.csv')
    output_file = os.path.join(output_directory, 'ALL_JOBS_SUMMARY_Merged.csv')

    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_file, encoding='utf-8-sig')

    # Headers for Job_Description and Qualifications
    job_desc_headers = [
        "Responsibilities:",
        "Overview:"
    ]

    qual_headers = [
        "Required Qualifications",
        "Preferred Qualifications",
        "Required/Minimum Qualifications",
        "Minimum Qualifications"
    ]

    # Apply the format_text function for Qualifications and Job_Description columns
    df['Qualifications_2'] = df['Qualifications'].apply(format_text, headers=qual_headers)
    df['Job_Description_2'] = df['Job_Description'].apply(format_text, headers=job_desc_headers)

    # Save the formatted DataFrame to a CSV file
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"CSV saved to {output_file}")

    # Define input and output files
    input_file = os.path.join(input_directory, 'ALL_JOBS_SUMMARY_Merged.csv')
    output_file = os.path.join(output_directory, 'ALL_JOBS_SUMMARY_Merged_Salary.csv')

    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_file, encoding='utf-8-sig')

    # Convert date posted and date closed to datetime objects
    if ('Date posted' in df.columns):
        df['Date posted'] = pd.to_datetime(df['Date posted']).fillna("")
    if('Closing date' in df.columns):
        df['Closing date'] = pd.to_datetime(df['Closing date']).fillna("")

    # Apply the calculate_hourly_pay function to the Salary Range column to create the Pay Range column
    df["Pay_Range"] = ""
    for (index, data) in df.iterrows():
     salary,pay_range = calculate_hourly_pay(data['Salary'])
     df.at[index,'Salary'] = salary
     df.at[index,'Pay_Range'] = pay_range

    # Save the updated DataFrame to a CSV file
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"CSV saved with Pay_Range in $/hour and Job Close Date to {output_file}")

    # Print "Salary_Range" and "Job Close Date"
    print(df[['Salary', 'Closing Date', 'Date posted']])

    # Define input file
    input_file = os.path.join(input_directory, 'ALL_JOBS_SUMMARY_Merged_Salary.csv')  # The CSV file must be in this directory

    # Load the CSV file into a DataFrame with 'utf-8-sig' encoding
    df = pd.read_csv(input_file, encoding='utf-8-sig')

    # Print column headers vertically
    print("Column headers in the file (printed vertically with 'utf-8-sig' encoding):")
    for column in df.columns:
        print(column)

    # Define input and output files
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
    df_output["Job Title"] = df_input["Job Title"]
    df_output["Job_AI_Summary"] = df_input["Job_AI_Summary"]
    df_output["Job Requisition Number"] = df_input["Job number"]
    df_output["Link to Apply"] = df_input["URL"]
    df_output["Job Type (Full, Part, Intern)"] = df_input["Employment type"]
    df_output["Compensation"] = df_input["Pay_Range"]
    df_output["Job Open Date"] = df_input["Date posted"]
    df_output["Job Close Date"] = df_input["Closing Date"]
    df_output["Company or Organization"] = "Microsoft"
    df_output["Business Unit / Division"] = df_input["Profession"]
    df_output["Job Category"] = df_input["Discipline"]
    df_output["Location"] = df_input["Location"]
    df_output["Qualifications"] = df_input["Qualifications_2"]
    df_output["Position Description"] = df_input["Job_Description_2"]
    df_output["Expected Salary"] = df_input["Salary"]

    # Save the output DataFrame with all the headers to the specified output file
    df_output.to_csv(output_file, index=False, encoding='utf-8-sig')


    print(f"CSV saved to {output_file}")

if __name__ == "__main__":
    main()