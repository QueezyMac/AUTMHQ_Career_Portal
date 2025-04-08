import MSFT
import CityofAustin
import emerson
import multiprocessing

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