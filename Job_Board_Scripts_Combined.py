import MSFT
import CityofAustin
import threading

if __name__ == "__main__":
    # Run script to extract the current jobs posted on the City of Austin's job board
    CityofAustin.main()
    # Run script to extract the current jobs posted on Microsoft's job board
    MSFT.main()