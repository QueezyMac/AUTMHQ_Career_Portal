import MSFT
import CityofAustin
import multiprocessing

if __name__ == "__main__":
    # Create a process to extract the current jobs posted on the City of Austin's job board
    p1 = multiprocessing.Process(target=CityofAustin.main)
    # Create a process to extract the current jobs posted on Microsoft's job board
    p2 = multiprocessing.Process(target=MSFT.main)


    # Start the process to extract the current jobs posted on the City of Austin's job board
    p1.start()
    # Start the process to extract the current jobs posted on Microsoft's job board runs at the
    # same time as the process to extract the current jobs posted on the City of Austin's job board
    p2.start()
    
    # Wait until processes finish to run the code after these two processes finish executing
    p1.join()
    p2.join()