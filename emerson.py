from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from bs4 import BeautifulSoup
import openai
import os
import pandas as pd
import re


# Function to get the full HTML code and urls for each job
def get_job_sources(url):
    driver = webdriver.Chrome()
    # Open the Emerson job board showing all the jobs in Austin, TX
    driver.get(url)
    # Wait 5 seconds to load the full page
    time.sleep(5)

    # When the page opens, you have to click on a button to accept all the cookies
    accept_cookies_button = driver.find_element(By.XPATH,"//button[text()='Accept All Cookies']")
    accept_cookies_button.click()
    time.sleep(5)
    
    # Find the number of open jobs and convert to an integer
    jobs_open = driver.find_element(By.XPATH,"//h2[@class='search-jobs__counter text-color-secondary']")
    jobs_open = jobs_open.get_attribute("innerText").split()[0]
    jobs_open = int(jobs_open)


    # Ensure all the jobs from Emerson's job board are loaded
    links = []
    while (len(links) < jobs_open):
        driver.execute_script("window.scrollBy(0,document.body.scrollHeight);")
        time.sleep(2)
        links = driver.find_elements(By.XPATH,"//a[@class= 'job-list-item__link']")

    # Get the HTML code and URL's for each job
    page_sources = []
    urls = []
    for link in links:
        link.click()
        time.sleep(3)
        urls.append(driver.current_url)
        page_sources.append(driver.page_source)
        driver.back()

    driver.quit()
    return(urls, page_sources)


# Function to test if it is a heading
def is_heading(value, string):
    lines = string.split("\n")
    for line in lines:
        if(line==value+":" or line==value):
            return(True)
    return(False)

# Function to find where a sub string in the body of the job description is a heading
def find_heading(value, key, string, start_or_end):
    # It is a heading if it is in the format of \n{value}\n or \n{value}:\n or the heading is the overview,
    # and the string is formatted as {value}\n or {value}:\n or the value is blank. This is because the value can be blank
    # for the overview section
    if(key == "Overview"):
        match = re.search(re.compile(re.escape(value)+r':?\n',re.IGNORECASE),string)
    else:
        match = re.search(re.compile(r'\n'+re.escape(value)+r':?\n',re.IGNORECASE),string)

    # Check if the key is overview or if it is finding the start of the section or the end of the section
    if(key == "Overview" and value=="" and start_or_end=="start"):
        return(0)
    elif(match and start_or_end=="start"):
        return(match.span()[1])
    elif(match and start_or_end=="end"):
        return(match.span()[0])
    else:
        return(len(string))

# Function to get the information for one job
def get_job_info(page_source, url, file_number, directory="EMERSON_JOBS"):
    # Dictionary that stores the job information
    job_info = {}

    # Store URL and file number
    job_info["URL"] = url
    job_info["CSV"] = "'{:03}".format(file_number)
    # Extract job title
    soup = BeautifulSoup(page_source, "html.parser")
    job_info["Job Title"] = (soup.select_one("h1.job-details__title").get_text().replace("\\n","\n")
                             .replace("\u200b","").replace("\r","").replace("\xa0"," ").strip())
    # Extract Location
    location = soup.select_one("div.job-details__subtitle")
    job_info["Location"] = (location.get_text().replace("\\n","\n").replace("\u200b","").replace("\r","")
                            .replace("\xa0"," ").strip())
    # Clean up location text
    location_lines = [line for line in job_info["Location"].split("\n") if line.strip()!=""]
    job_info["Location"] = ""
    for line in location_lines:
        job_info["Location"] = job_info["Location"]+" "+line.strip()
    job_info["Location"] = job_info["Location"].replace(" ","",1)

    # Get job information from the bottom of the page
    info = soup.select("div.job-details__info-section li")
    for item in info:
        [key,value] = (item.get_text("\n").replace("\\n","\n").replace("\u200b","").replace("\r","").replace("\xa0"," ")
                       .strip().split("\n",1))
        if(key != "Locations"):
            job_info[key] = value.strip()

    # Job schedule tells the Job type (Full, Part, Intern, Co-op)
    # Therefore, update job type to intern or co-op if it should be that
    if("intern" in job_info["Job Title"].lower()):
        job_info["Job Schedule"] = "Internship"
    elif("co-op" in job_info["Job Title"].lower()):
        job_info["Job Schedule"] = "Co-op"
    
    # Get job description and qualification and ensure there are no empty lines
    job_description = soup.select_one("div.job-details__description-content")

    # Clean up text by removing span tags, a tags, strong tags, i tags, sup tags, sub tags, and b tags
    span_tags = job_description.find_all("span")
    for tag in span_tags:
        tag.unwrap()

    a_tags = job_description.find_all("a")
    for tag in a_tags:
        tag.unwrap()

    strong_tags = job_description.find_all("strong")
    for tag in strong_tags:
        tag.unwrap()

    i_tags = job_description.find_all("i")
    for tag in i_tags:
        tag.unwrap()

    sup_tags = job_description.find_all("sup")
    for tag in sup_tags:
        tag.unwrap()

    sub_tags = job_description.find_all("sub")
    for tag in sub_tags:
        tag.unwrap()

    b_tags = job_description.find_all("b")
    for tag in b_tags:
        tag.unwrap()
    
    job_description = BeautifulSoup(str(job_description),"html.parser")
    
    # Clean up the job description and qualifications by removing empty lines
    job_description = job_description.get_text("\n").strip()
    job_description_list = [line.replace("\\n","").replace("\u200b","").replace("\r","").replace("\xa0"," ").replace("\u202f"," ")
                            .strip() for line in job_description.split("\n") if line!=""]
    job_description=""
    for line in job_description_list:
        if(line.strip() != ""):
            job_description = job_description+line.replace("\n","")+"\n"

    # Ensure job description and qualifications are in a standard format
    job_description = job_description.replace(":\n",": ")
    job_description = job_description.replace(": ",":\n")

    # Dictionary with markers that find the sections in the body of the job description.
    # Keys are the headings on the CSV files generated.
    # The contents in the dictionary are the start markers list and the end markers list.
    # The headings for the sections on the page are different for each job description, so the headings are stored as list.
    markers = {
        "Overview": (["Position Overview", "Summary", "Job Summary", "Objective of Role", "Job Description"],
                     ["Organization", "In this Role, Your Responsibilities Will Be", "In This Role, Your Responsibilities Will",
                      "Key Responsibilities", "Primary Responsibilities", "Responsibilities", "Key Skills and Competencies",
                      "Core Job Responsibilities","Core Responsibilities","For This Role, You Will Need"]),
        "Organization": (["Organization"],["In this Role, Your Responsibilities Will Be",
                                           "In this Role, Your Responsibilities Will", "Key Responsibilities",
                                           "Primary Responsibilities", "Responsibilities"]),
        "Responsibilities": (["In this Role, Your Responsibilities Will Be","In this Role, Your Responsibilities Will",
                              "Key Responsibilities", "Primary Responsibilities", "Responsibilities", "Core Job Responsibilities",
                              "Core Responsibilities"],
                             ["Who You Are", "This job might be for you if", "For This Role, You Will Need", "For This Role, Will Need",
                              "For This Role You Will Need", "Required Qualifications", "Required Qualification", "Requirements",
                              "Basic Requirements", "Qualifications", "Minimum Requirements", "Skills", "Expectations",
                              "Our Culture & Commitment to You"]),
        "Required Qualifications": (["For This Role, You Will Need", "For This Role, Will Need",
                                     "The following skills and experience are required", "For This Role You Will Need",
                                     "Key Skills and Competencies", "Required Qualifications", "Required Qualification",
                                     "Requirements", "Basic Requirements", "Minimum Requirements", "Qualifications"],
                                    ["Preferred Qualifications that Set You Apart", "Preferred Qualifications", "Preferred Skills",
                                     "Preferred Requirements","Expectations","Who You Are", "This job might be for you if",
                                     "Demonstrated ability to", "Our Offer To You", "Our Culture & Commitment to You"]),
        "Skills": (["Skills"],["Demonstrated ability to"]),
        "Preferred Qualifications": (["Preferred Qualifications that Set You Apart", "Preferred Qualifications", "Preferred Skills",
                                      "Preferred Requirements"],
                                     ["Who You Are", "This job might be for you if", "Demonstrated ability to",
                                      "Our Culture & Commitment to You", "Our Offer To You", "Our Perks", "At Emerson"]),
        "Expectations": (["Expectations"], ["Our Culture & Commitment to You"]),
        "Who You Are": (["Who You Are", "This job might be for you if", "Demonstrated ability to"],
                        ["For This Role, You Will Need", "For This Role You Will Need", "Basic Requirements",
                         "The following skills and experience are required", "Qualifications",
                         "Our Culture & Commitment to You"])
    }

    # Iterate through markers dictionary to extract the overview, resposibilities, required qualifications,
    # preferred qualifications, and who you are section
    for key,(start_list, end_list) in markers.items():
        # Finds the start marker in the job description for the current key it is on
        for value in start_list:
            if(value.upper() in job_description.upper() and is_heading(value.upper(),job_description.upper())):
                start = value
                break
            else:
                start = ""
        # Loop through end markers
        new_end_list = []
        for value in end_list:
            # Find the location of all the headings after the start index because the sections are not always in the same order
            if(value.upper() in job_description.upper() and (is_heading(value.upper(),job_description.upper())
               and (key == "Overview" or
               (find_heading(value,key,job_description,"end")+1>=find_heading(start,key,job_description,"start") and
               find_heading(value,key,job_description,"end") != len(job_description))) or value == "At Emerson")):
                new_end_list.append(find_heading(value,key,job_description,"end"))
            # If list is empty and the end marker is on At Emerson, find the end index
            elif(value == "At Emerson" and len(new_end_list)==0):
                end_index = job_description.find("\nAt Emerson")+len("\nAt Emerson")
            # Otherwise, the section finishes at the end of the string
            else:
                end_index = len(job_description)
        
        # If the start marker is blank and the section is not the overview,
        # it means the section is not in the job description. Therefore, set the start_index to 0
        if(start==""):
            start_index = 0
        else:
            start_index = find_heading(start,key,job_description,"start")
        
        # If the list is not empty, sort the list and get the lowest end index because that is where the section ends
        if(len(new_end_list)!=0):
            new_end_list.sort()
            end_index = new_end_list[0]

        # If the start index is 0 and the key is not Overview, the section is not in the job description
        if(start_index==0 and key != "Overview"):
            if(job_info.get(key) == None):
                job_info[key] = ""
        else:
            job_info[key] = job_description[start_index:end_index]

    # Get business unit/Division
    job_info["Business Unit / Division"] = ""
    if(("TEST & MEASUREMENT" in job_description.upper()) or ("TEST AND MEASUREMENT" in job_description.upper())):
        job_info["Business Unit / Division"] = "Test & Measurement"
    elif("AUTOMATION" in job_description.upper()):
        job_info["Business Unit / Division"] = "Automation Solutions"
    elif(("COMMERCIAL & RESIDENCY SOLUTIONS" in job_description.upper()) or 
         ("COMMERCIAL AND RESIDENCY SOLUTIONS" in job_description.upper())):
        job_info["Business Unit / Division"] = "Commercial & Residency Solutions"
    elif("PROCESS MANAGEMENT" in job_description.upper()):
        job_info["Business Unit / Division"] = "Process Management"
    elif("INDUSTRIAL SOLUTIONS" in job_description.upper()):
        job_info["Business Unit / Division"] = "Industrial Solutions"

    # Get job salary if it is posted for the job, otherwise leave it blank.
    salary_start = "The salary range for this role is "
    salary_end = "Annually"
    section_start = "At Emerson"
    job_info["Salary"] = ""
    if(section_start.upper() in job_description.upper()):
        start_index = job_description.upper().find(section_start.upper())
        section = job_description[start_index:]
        if(salary_start.upper() in section.upper()):
            start_index = section.upper().find(salary_start.upper())+len(salary_start)
            end_index = section.upper().find(salary_end.upper(),start_index)
            job_info["Salary"] = section[start_index:end_index] + "per year"

    # Save the job information to a CSV file
    job_info_df = pd.DataFrame([job_info])
    file = os.path.join(directory, "{:03}.csv".format(file_number))
    job_info_df.to_csv(file, index=False)

# Function to get the information from all the jobs
def get_all_job_info(page_sources, urls, directory="EMERSON_JOBS"):
    count = 0
    while(count<len(page_sources) and count<len(urls)):
        get_job_info(page_sources[count],urls[count],count,directory)
        count = count+1

# Function to get ALL_JOBS.csv file
def get_all_jobs_csv(directory="EMERSON_JOBS"):
    count = 0

    file = os.path.join(directory,"{:03}.csv".format(count))
    jobs_info = pd.DataFrame()
    while(os.path.exists(file)):
        job_info = pd.read_csv(file)
        # Emerson sometimes posts the same job multiple times, so this only stores the jobs in the ALL_JOBS.csv file once
        try:
            if(jobs_info.loc[jobs_info["Job Identification"]==job_info["Job Identification"]].empty):
                jobs_info = pd.concat([jobs_info,job_info], ignore_index=True)
        except (KeyError, ValueError) as e:
            jobs_info = pd.concat([jobs_info,job_info], ignore_index=True)

        count = count+1
        file = os.path.join(directory,"{:03}.csv".format(count))

    # Merge required qualifications, preferred qualifications, expectations, and who you are into a column labeled "Qualifications"
    jobs_info["Qualifications"] = ""
    # Merge overview and responsibilities into a column labeled "Job Description"
    jobs_info["Job Description"] = ""

    # Loop through the rows in the dataframe to add the qualifications and job description
    for (index, data) in jobs_info.iterrows():
        qualification_categories_exist = {"Required Qualifications": not(pd.isna(data["Required Qualifications"])),
                                          "Skills": not(pd.isna(data["Skills"])),
                                          "Preferred Qualifications": not(pd.isna(data["Preferred Qualifications"])),
                                          "Expectations": not(pd.isna(data["Expectations"])),
                                          "Who You Are": not(pd.isna(data["Who You Are"]))}

        # Some of the sections for qualifications are blank, so only put the corresponding heading if the section is not blank
        for (category, category_exists) in qualification_categories_exist.items():
            if(category_exists):
                jobs_info.at[index,"Qualifications"] = (jobs_info.at[index,"Qualifications"]+
                                                        "\n\n"+category+":\n"+data[category])
        # The first heading should not have new line characters
        jobs_info.at[index,"Qualifications"] = jobs_info.at[index,"Qualifications"].replace("\n\n","",1)


        job_description_categories_exist = {"Overview": not(pd.isna(data["Overview"])),
                                            "Organization": not(pd.isna(data["Organization"])),
                                            "Responsibilities": not(pd.isna(data["Responsibilities"]))}
    
        # Some of the sections for the job description are blank, so only put the heading if the section is not blank
        for (category, category_exists) in job_description_categories_exist.items():
            if(category_exists):
                jobs_info.at[index,"Job Description"] = (jobs_info.at[index,"Job Description"]+
                                                        "\n\n"+category+":\n"+data[category])
        # The first heading should not have new line characters
        jobs_info.at[index,"Job Description"] = jobs_info.at[index,"Job Description"].replace("\n\n","",1)

    # Organize columns
    column_list = ["URL","CSV","Job Title","Location","Posting Date","Apply Before","Job Identification",
                   "Business Unit / Division", "Job Function","Job Schedule","Qualifications","Job Description",
                   "Salary"]
    for column in column_list:
        if(not(column in jobs_info.columns)):
            jobs_info[column] = ""
    jobs_info = jobs_info[column_list]
    
    file = os.path.join(directory,"ALL_JOBS.csv")
    jobs_info.to_csv(file, index=False)


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
    folder_path = os.path.join(os.getcwd(), "EMERSON_JOBS")
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
                df_summary_job_description_no_blanks = re.sub(r"\s+","",df_summary.loc[df_summary['Job Identification'] 
                                                                                       == row['Job Identification'],
                                                                                       'Job Description'].item())
                df_job_description_no_blanks = re.sub(r"\s+","",row["Job Description"])
                if(df_summary_job_description_no_blanks == df_job_description_no_blanks):
                    summary = df_summary.loc[df_summary['Job Identification'] == row['Job Identification'],
                                'Job_AI_Summary'].item()
                    print(f"CSV {row['CSV']} already generated")
                else:
                    summary = prompt_openai(row['Job Description'])
            except ValueError:
                summary = prompt_openai(row['Job Description'])
            df.at[index, 'Job_AI_Summary'] = summary
            print(f"CSV {row['CSV']} : {summary}\n")

        output_path = os.path.join(folder_path, 'ALL_JOBS_SUMMARY.csv')
        df.to_csv(output_path, index=False, encoding='utf-8-sig')  # Use the specified encoding
        print(f"New CSV exported to {output_path}")
    else:
        for index, row in df.iterrows():
            summary = prompt_openai(row['Job Description'])
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
    
    numbers = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20"]
    # For each line, strip whitespace, then check if it starts with any of the headers or is already formatted.
    # If not, and if the line is not empty, add "- " in front of it
    formatted_lines = [
        line if any(line.strip().startswith(header) for header in headers) or line.strip() == "" or
        any(line.strip().startswith(number+".") for number in numbers) or line.strip().startswith("-")
        or line.strip().startswith("·") else "- " + line for line in lines
    ]
    
    # Join the formatted lines and return
    return "\n".join(formatted_lines)

# Define a function to calculate the hourly Pay Range based on the Salary Range
def calculate_hourly_pay(salary_range):
    # Check if salary_range is NaN, if so, return "DOE"
    if pd.isna(salary_range):
        return "DOE","DOE"
    # Remove " per year" for calculation
    salary_range = salary_range.replace(" per year", "")
    # Extract minimum and maximum values from the salary range
    try:
        min_salary, max_salary = [float(val.replace("$", "").replace(",", "").strip()) for val in salary_range.split('-')]
        hourly_low = min_salary / 2080
        hourly_high = max_salary / 2080
        return ("${:,} – ${:,} per year".format(int(min_salary),int(max_salary)),
                "${:.2f} – ${:.2f} per hour".format(hourly_low, hourly_high))
    except:
        return "DOE","DOE"


def main():
    # Setup OpenAI API key
    api_key = os.getenv("api_key")
    openai.api_key = api_key

    # Make the EMERSON_JOBS folder if it has not been created yet
    if(not os.path.exists("EMERSON_JOBS")):
        os.makedirs("EMERSON_JOBS")

    # Remove all the files in the directory besides the ALL_JOBS_SUMMARY.csv file
    for file in os.listdir("EMERSON_JOBS"):
        if(file != "ALL_JOBS_SUMMARY.csv"):
            os.remove(os.path.join("EMERSON_JOBS",file))

    # Link to the Emerson job board showing all the jobs in Austin, TX
    url = "https://hdjq.fa.us2.oraclecloud.com/hcmUI/CandidateExperience/en/sites/CX_1/jobs?location=Austin%2C+TX%2C+United+States&locationId=300000002015186&locationLevel=city&mode=location&radius=25&radiusUnit=MI"

    # Get full HTML code and urls for each job listed on the link above
    [urls, page_sources] = get_job_sources(url)

    # Get the job information for all the jobs listed on the link at the top of this function
    get_all_job_info(page_sources, urls)

    # Generate ALL_JOBS.csv file which contains the job information for all the jobs from the link at the top of this function
    get_all_jobs_csv()


    #  OPEN AI JOB SUMMARY

    # Call the generate_summaries function
    generate_summaries()


    # PYTHON CLEANUP & Addition of markers: '-'

    # Define input and output directories and files
    input_directory = "EMERSON_JOBS"
    output_directory = "EMERSON_JOBS"
    input_file = os.path.join(input_directory, 'ALL_JOBS_SUMMARY.csv')
    output_file = os.path.join(output_directory, 'ALL_JOBS_SUMMARY_Merged.csv')

    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_file, encoding='utf-8-sig')

    # Headers for Job_Description and Qualifications
    job_desc_headers = [
        "Overview:",
        "Organization:",
        "Responsibilities:"        
    ]

    qual_headers = [
        "Required Qualifications:",
        "Preferred Qualifications:",
        "Skills:",
        "Expectations:",
        "Who You Are:"
    ]

    # Apply the format_text function for Qualifications and Job_Description columns
    df['Qualifications_2'] = df['Qualifications'].apply(format_text, headers=qual_headers)
    df['Job_Description_2'] = df['Job Description'].apply(format_text, headers=job_desc_headers)

    # Save the formatted DataFrame to a CSV file
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"CSV saved to {output_file}")


    # PYTHON Salary calculation, date format & Addition of markers: '-'

    # Define input and output directories and files
    input_directory = "EMERSON_JOBS"
    output_directory = "EMERSON_JOBS"
    input_file = os.path.join(input_directory, 'ALL_JOBS_SUMMARY_Merged.csv')
    output_file = os.path.join(output_directory, 'ALL_JOBS_SUMMARY_Merged_Salary.csv')

    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_file, encoding='utf-8-sig')

    # Ensure "Posting Date" and "Apply Before" is in YYYY-MM-DD format
    if 'Posting Date' and 'Apply Before' in df.columns:
        df['Posting Date'] = pd.to_datetime(df['Posting Date'], format="%m/%d/%Y, %H:%M %p")
        df['Posting Date'] = df['Posting Date'].dt.strftime("%Y-%m-%d")
        df['Apply Before'] = pd.to_datetime(df['Apply Before'], format="%m/%d/%Y, %H:%M %p")
        df['Apply Before'] = df['Apply Before'].dt.strftime("%Y-%m-%d")

    # Apply the calculate_hourly_pay function to the Salary Range column to create the Pay Range column
    df["Pay_Range"] = ""
    if 'Salary' in df.columns:
        for (index,data) in df.iterrows():
            df.at[index,"Salary"], df.at[index,"Pay_Range"] = calculate_hourly_pay(data["Salary"])

    # Save the updated DataFrame to a CSV file
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    if 'Salary' in df.columns:
        print(f"CSV saved with Pay_Range in $/hour and Job Close Date to {output_file}")

        # Print "Salary_Range" and "Job Close Date"
        print(df[['Salary', 'Apply Before', 'Posting Date']])

    # Define input directory and file
    input_directory = "EMERSON_JOBS"  # Make sure this directory exists in your environment
    input_file = os.path.join(input_directory, 'ALL_JOBS_SUMMARY_Merged_Salary.csv')  # The CSV file must be in this directory

    # Load the CSV file into a DataFrame with 'utf-8-sig' encoding
    df = pd.read_csv(input_file, encoding='utf-8-sig')

    # Print column headers vertically
    print("Column headers in the file (printed vertically with 'utf-8-sig' encoding):")
    for column in df.columns:
        print(column)


    # MAP THE OUTPUT & PREPARE FOR WIX UPLOAD ! ! !

    # Define input and output directories and files
    input_directory = "EMERSON_JOBS"
    output_directory = "EMERSON_JOBS"
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
    df_output["Job Title"] = df_input["Job Title"]
    df_output["Job_AI_Summary"] = df_input["Job_AI_Summary"]
    df_output["Job Requisition Number"] = df_input["Job Identification"]
    df_output["Link to Apply"] = df_input["URL"]
    df_output["Job Type (Full, Part, Intern, Co-op)"] = df_input["Job Schedule"]
    df_output["Compensation"] = df_input["Pay_Range"]
    df_output["Job Open Date"] = df_input["Posting Date"]
    df_output["Job Close Date"] = df_input["Apply Before"]
    df_output["Company or Organization"] = "Emerson"
    df_output["Business Unit / Division"] = df_input["Business Unit / Division"]
    df_output["Job Category"] = "Emerson"
    df_output["Qualifications"] = df_input["Qualifications_2"]
    df_output["Position Description"] = df_input["Job_Description_2"]
    df_output["Location"] = df_input["Location"]
    df_output["Expected Salary"] = df_input["Salary"]

    # Save the output DataFrame with all the headers to the specified output file
    df_output.to_csv(output_file, index=False, encoding='utf-8-sig')


    print(f"CSV saved to {output_file}")


if __name__ == "__main__":
    main()