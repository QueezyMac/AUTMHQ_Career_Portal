import MSFT
import CityofAustin
import emerson
import multiprocessing
import os
import pandas as pd

# Function to combine the csv files that contain the jobs posted on the job boards for each company
def combine_csv_files(file_list=[os.path.join("Job Posts by URL", "ALL_JOBS_Ready-2-Upload-2-WIX.csv"),
                                 os.path.join("MSFT_JOBS", "ALL_JOBS_Ready-2-Upload-2-WIX.csv"),
                                 os.path.join("EMERSON_JOBS", "ALL_JOBS_Ready-2-Upload-2-WIX.csv")]):
    # Combine the data from each csv file to one dataframe
    df = pd.DataFrame()
    for file in file_list:
        if os.path.exists(file):
            new_df = pd.read_csv(file)
            df = pd.concat([df,new_df], ignore_index=True)
    
    # If the directory that stores the combined CSV file does not exist, create a new folder
    folder = "JOBS_FROM_ALL_COMPANIES"
    file = os.path.join(folder,"ALL_JOBS_Ready-2-Upload-2-WIX.csv")
    if(not os.path.exists(folder)):
        os.makedirs(folder)
    # Otherwise, if the file exists already delete it
    elif(os.path.exists(file)):
        os.remove(file)
    
    # Convert the dataframe to a CSV file
    df.to_csv(file,index=False)

if __name__ == "__main__":
    # Create a process to extract the current jobs posted on the City of Austin's job board
    p1 = multiprocessing.Process(target=CityofAustin.main)
    # Create a process to extract the current jobs posted on Microsoft's job board that are in Austin, TX
    p2 = multiprocessing.Process(target=MSFT.main)
    # Create a process to extract the current jobs ponsted on Emerson's job board that are in Austin, TX
    p3 = multiprocessing.Process(target=emerson.main)


    # Start processes to extract jobs from each of the job boards with the filters, if applicable, at the same time
    p1.start()
    p2.start()
    p3.start()
    
    # Wait until processes finish to run the code after these processes finish executing
    p1.join()
    p2.join()
    p3.join()

    # Call the function to combine the CSV files that get uploaded to the website to one CSV file
    combine_csv_files()