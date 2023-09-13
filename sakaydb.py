"""
SakayDB
A module for managing ride-hailing data.
A final project for Programming for Data Science
Author:
    Joshua Victor San Juan
    Jeremiah Dominic Soliman
    MSDS 2024
"""


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


class SakayDBError(ValueError):
    """
    Custom exception for SakayDB errors.
    
    Parameters
    ----------
    error_prompt : str, optional
        Description of the error. Default is None.
    """

    def __init__(self, error_prompt=None):
        """Initialize with an optional error description."""
        super().__init__(error_prompt)


class SakayDB:
    """A class to manage a database for Sakay trips.

    Attributes
    ----------
    data_dir : str
        The directory where the database files are stored.

    Methods
    -------
    __init__(data_dir: str)
        Initializes the SakayDB object with the specified data directory.

    add_trip(driver: str, pickup_datetime: str, dropoff_datetime: str,
             passenger_count: int, pickup_loc_name: str, dropoff_loc_name: str,
             trip_distance: float, fare_amount: float) -> int
        Adds a new trip to the database and returns the trip ID.

    add_trips(trips: List[Dict]) -> List[int]
        Adds multiple trips to the database and returns a list of trip IDs.

    delete_trip(trip_id: int)
        Deletes a trip from the database based on the trip ID.

    search_trips(**kwargs) -> pd.DataFrame
        Searches for trips in the database based on various criteria.

    export_data() -> pd.DataFrame
        Exports the trip data as a DataFrame.

    generate_statistics(stat: str) -> Dict
        Generates statistics based on the specified type ('trip', 'passenger',
        'driver', 'all').

    plot_statistics(stat: str)
        Plots statistics based on the specified type ('trip', 'passenger',
        'driver').

    generate_odmatrix(date_range: Optional[Tuple[str, str]]) -> pd.DataFrame
        Generates an origin-destination matrix for the specified date range.

    Exceptions
    ----------
    SakayDBError
        Custom exception for handling SakayDB errors.
    """

    def __init__(self, data_dir):
        """Initialize the SakayDB object.

        Parameters
        ----------
        data_dir : str
            The directory where the CSV files are stored.
        """
        self.data_dir = data_dir

    def add_trip(
        self,
        driver,
        pickup_datetime,
        dropoff_datetime,
        passenger_count,
        pickup_loc_name,
        dropoff_loc_name,
        trip_distance,
        fare_amount,
    ):
        """Add a new trip to the SakayDB database.

        Parameters
        ----------
        driver : str
            The name of the driver in the format "Last Name, Given Name".
        pickup_datetime : str
            The pickup date and time in the format 'HH:MM:SS,DD-MM-YYYY'.
        dropoff_datetime : str
            The drop-off date and time in the format 'HH:MM:SS,DD-MM-YYYY'.
        passenger_count : int
            The number of passengers for the trip.
        pickup_loc_name : str
            The name of the pickup location.
        dropoff_loc_name : str
            The name of the drop-off location.
        trip_distance : float
            The distance of the trip in miles.
        fare_amount : float
            The fare amount for the trip.

        Returns
        -------
        int
            The ID of the newly added trip.

        Raises
        ------
        SakayDBError
            If any of the input parameters have invalid or incomplete
            information.
        """
        # Assigning and cleaning input
        try:
            driver_name_list = driver.strip().split(", ")
            last_name = driver_name_list[0]
            given_name = driver_name_list[1]
            if len(driver_name_list) != 2:
                raise SakayDBError("has invalid or incomplete information")
            pickup_datetime = pickup_datetime.strip()
            pickup_date = pd.to_datetime(pickup_datetime,
                                         format="%H:%M:%S,%d-%m-%Y")
            dropoff_datetime = dropoff_datetime.strip()
            dropoff_date = pd.to_datetime(dropoff_datetime,
                                          format="%H:%M:%S,%d-%m-%Y")
            pickup_loc_name = pickup_loc_name.strip()
            dropoff_loc_name = dropoff_loc_name.strip().title()
        except Exception:
            raise SakayDBError("has invalid or incomplete information")
        # Path creation
        trips_path = os.path.join(self.data_dir, "trips.csv")
        drivers_path = os.path.join(self.data_dir, "drivers.csv")
        locations_path = os.path.join(self.data_dir, "locations.csv")

        # Function to read or create csv file
        def read_or_create_csv(path, columns):
            """Read a CSV file from the given path or create a new DataFrame
            with the specified columns if the file is not found.

            Parameters
            ----------
            path : str
                The path to the CSV file to read.
            columns : list
                The column names for the DataFrame to create if the CSV file is
                not found.

            Returns
            -------
            pd.DataFrame
                The DataFrame read from the CSV file or a new DataFrame with
                the specified columns.

            Raises
            ------
            FileNotFoundError
                If the CSV file is not found and a new DataFrame is created
                instead.
            """
            try:
                return pd.read_csv(path)
            except FileNotFoundError:
                return pd.DataFrame(columns=columns)

        # Columns for CSV file
        trips = read_or_create_csv(
            trips_path,
            [
                "trip_id",
                "driver_id",
                "pickup_datetime",
                "dropoff_datetime",
                "passenger_count",
                "pickup_loc_id",
                "dropoff_loc_id",
                "trip_distance",
                "fare_amount",
            ],
        )

        drivers = read_or_create_csv(
            drivers_path, ["driver_id", "given_name", "last_name"]
        )

        locations = read_or_create_csv(locations_path,
                                       ["location_id", "loc_name"])

        # Check if driver in database, else create new entry.
        if drivers.shape[0] == 0:
            driver_id = 1
            new_driver = pd.DataFrame(
                {
                    "driver_id": [driver_id],
                    "given_name": [given_name],
                    "last_name": [last_name],
                }
            )
            drivers = pd.concat([drivers, new_driver], ignore_index=True)
        else:
            matching_drivers = drivers[
                (drivers["given_name"].str.casefold() == given_name.casefold())
                & (drivers["last_name"].str.casefold() == last_name.casefold())
            ]
            if matching_drivers.shape[0] > 0:
                driver_id = matching_drivers["driver_id"].values[0]
            else:
                driver_id = drivers["driver_id"].iloc[-1] + 1
                new_driver = pd.DataFrame(
                    {
                        "driver_id": [driver_id],
                        "given_name": [given_name],
                        "last_name": [last_name],
                    }
                )
                drivers = pd.concat([drivers, new_driver], ignore_index=True)

        # Handle location information
        if locations.shape[0] == 0:
            pickup_loc_id = 1
            dropoff_loc_id = 1
        else:
            # Handle pickup location ID
            if pickup_loc_name in locations["loc_name"].values.tolist():
                pickup_loc_id = locations[
                    locations["loc_name"] == pickup_loc_name
                ]["location_id"].values[0]
            else:
                pickup_loc_id = locations["location_id"].iloc[-1] + 1
                new_row = pd.DataFrame(
                    {"location_id": [pickup_loc_id],
                        "loc_name": [pickup_loc_name]}
                )
                locations = pd.concat([locations, new_row], ignore_index=True)

            # Handle dropoff location ID
            if dropoff_loc_name in locations["loc_name"].values.tolist():
                dropoff_loc_id = locations[
                    locations["loc_name"] == dropoff_loc_name
                ]["location_id"].values[0]
            else:
                dropoff_loc_id = locations["location_id"].iloc[-1] + 1
                new_row = pd.DataFrame(
                    {"location_id": [dropoff_loc_id],
                        "loc_name": [dropoff_loc_name]}
                )
                locations = pd.concat([locations, new_row], ignore_index=True)
        # New row to be added
        row = {
            "driver_id": int(driver_id),
            "pickup_datetime": pickup_datetime,
            "dropoff_datetime": dropoff_datetime,
            "passenger_count": int(passenger_count),
            "pickup_loc_id": int(pickup_loc_id),
            "dropoff_loc_id": int(dropoff_loc_id),
            "trip_distance": float(trip_distance),
            "fare_amount": float(fare_amount),
        }
        # Concatenating if not existing, error if in database
        try:
            if trips.shape[0] == 0:
                row["trip_id"] = 1
            elif row in (
                trips.loc[:, trips.columns != "trip_id"].to_dict(
                    orient="records")
            ):
                raise SakayDBError("is already in the database")
            else:
                row["trip_id"] = int(trips["trip_id"].iloc[-1] + 1)
            trips = pd.concat(
                [trips, pd.DataFrame.from_dict([row])], ignore_index=True)
        except SakayDBError:
            raise SakayDBError("is already in the database")

        # Creating CSV files after concatenating.
        trips.to_csv(os.path.join(self.data_dir, "trips.csv"), index=False)
        drivers.to_csv(os.path.join(self.data_dir, "drivers.csv"), index=False)
        locations.to_csv(os.path.join(
            self.data_dir, "locations.csv"), index=False)
        # Return trip id added
        return trips["trip_id"].iloc[-1]

    def add_trips(self, trips):
        """Add multiple trips to the database.

        Parameters
        ----------
        trips : list of dict
            A list of dictionaries, each containing the following keys:
            - 'driver': str, The full name of the driver, formatted as
                "Last Name, Given Name".
            - 'pickup_datetime': str, The pickup date and time in the format
                'HH:MM:SS,DD-MM-YYYY'.
            - 'dropoff_datetime': str, The dropoff date and time in the format
                'HH:MM:SS,DD-MM-YYYY'.
            - 'passenger_count': int, The number of passengers.
            - 'pickup_loc_name': str, The name of the pickup location.
            - 'dropoff_loc_name': str, The name of the dropoff location.
            - 'trip_distance': float, The distance of the trip in miles.
            - 'fare_amount': float, The fare amount for the trip.

        Returns
        -------
        list of int
            A list of trip IDs for the successfully added trips.

        Warnings
        --------
        - Prints a warning if a trip is already in the database.
        - Prints a warning if a trip has invalid or incomplete information.
        """
        trip_ids = []

        for i, trip in enumerate(trips):
            try:
                trip_ids.append(self.add_trip(**trip))

            except SakayDBError as e:
                print(f"Warning: trip index {i} {e}. Skipping...")
            except Exception:
                print(
                    f"Warning: trip index {i} has invalid or incomplete "
                    f"information. Skipping..."
                )

        return trip_ids

    def delete_trip(self, trip_id):
        """Delete a trip from the database based on the given trip ID.

        Parameters
        ----------
        trip_id : int
            The ID of the trip to be deleted.

        Raises
        ------
        SakayDBError
            - If the trips.csv file does not exist.
            - If the given trip ID is not found in the database.

        Notes
        -----
        This method modifies the trips.csv file to remove the specified trip.

        """
        trip_file_path = os.path.join(self.data_dir, "trips.csv")

        if not os.path.isfile(trip_file_path):
            raise SakayDBError
        else:
            df = pd.read_csv(trip_file_path)

        if trip_id not in df["trip_id"].values:
            raise SakayDBError
        else:
            df.drop(df.index[df["trip_id"] == trip_id], inplace=True)

        df.to_csv(trip_file_path, index=False)

    def search_trips(self, **kwargs):
        """Search for trips in the SakayDB database based on various criteria.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments specifying the search criteria.
            Valid keys include:
            - 'driver_id': int or tuple of ints
            - 'pickup_datetime': str or tuple of strs,
                format '%H:%M:%S,%d-%m-%Y'
            - 'dropoff_datetime': str or tuple of strs,
                format '%H:%M:%S,%d-%m-%Y'
            - 'passenger_count': int or tuple of ints
            - 'trip_distance': float or tuple of floats
            - 'fare_amount': float or tuple of floats

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the trips that match the search criteria.
            Returns an empty DataFrame if no trips match the criteria or if
            the database does not exist.

        Raises
        ------
        SakayDBError
            If any of the search criteria are invalid or incomplete.
        """
        # Keys for checking
        keys_complete = [
            "driver_id",
            "pickup_datetime",
            "dropoff_datetime",
            "passenger_count",
            "trip_distance",
            "fare_amount",
        ]
        keys_datetime = ["pickup_datetime", "dropoff_datetime"]
        keys_number = ["driver_id", "passenger_count",
                       "trip_distance", "fare_amount"]

        # Checks if valid keyword argument
        keys = kwargs.keys()
        if not set(kwargs.keys()).issubset(keys_complete) or not kwargs:
            raise SakayDBError

        # Check if the database exists; otherwise, return an empty list
        trip_path = os.path.join(self.data_dir, "trips.csv")

        df_trips = pd.read_csv(trip_path) if os.path.exists(
            trip_path) else None

        if df_trips is None:
            return []

        # Create string representations of datetime columns
        df_trips.insert(4, "pickup_string", df_trips["pickup_datetime"])
        df_trips.insert(5, "dropoff_string", df_trips["dropoff_datetime"])

        # Convert datetime columns to actual datetime format
        datetime_format = "%H:%M:%S,%d-%m-%Y"
        datetime_column = ["pickup_datetime", "dropoff_datetime"]
        df_trips[datetime_column] = df_trips[datetime_column].apply(
            pd.to_datetime, format=datetime_format
        )

        # For valid keys
        for key, value in kwargs.items():
            try:
                if isinstance(value, tuple):
                    if len(value) != 2:
                        raise SakayDBError

                    # Change value format to align with valid format
                    value = list(value)
                    datetime_format = "%H:%M:%S,%d-%m-%Y"

                    if key in keys_number:
                        value = [
                            float(v) if v is not None else None for v in value]

                    if key in keys_datetime:
                        value = [
                            pd.to_datetime(v, format=datetime_format)
                            if v is not None
                            else None
                            for v in value
                        ]

                    # Filter dataframe based on range
                    start, end = value

                    if start is not None and end is None:
                        df_trips = df_trips[df_trips[key] >= start]
                    elif start is None and end is not None:
                        df_trips = df_trips[df_trips[key] <= end]
                    elif start is not None and end is not None:
                        df_trips = df_trips[
                            (df_trips[key] >= start) & (df_trips[key] <= end)
                        ]

                else:
                    # Change value format to align with valid format
                    datetime_format = "%H:%M:%S,%d-%m-%Y"
                    if key in keys_number:
                        value = float(value)
                    if key in keys_datetime:
                        value = pd.to_datetime(value, format=datetime_format)

                    # Filter dataframe
                    df_trips = df_trips[(df_trips[key] == value)]

            except Exception:
                raise SakayDBError

        # Revert to original datetime (string format)
        column_mapping = {
            "pickup_string": "pickup_datetime",
            "dropoff_string": "dropoff_datetime",
        }
        # Sort, drop and rename dataframe
        df_trips = df_trips.sort_values(by=list(keys))
        df_trips.drop(columns=keys_datetime, inplace=True)
        df_trips.rename(columns=column_mapping, inplace=True)
        df_trips = df_trips.astype(
            {
                "driver_id": "int",
                "pickup_datetime": "str",
                "dropoff_datetime": "str",
                "passenger_count": "int",
                "trip_distance": "float",
                "fare_amount": "float",
            }
        )

        return df_trips

    def export_data(self):
        """Export the SakayDB data into a formatted DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the exported data with the following
            columns:
            - 'driver_lastname': str
            - 'driver_givenname': str
            - 'pickup_datetime': str
            - 'dropoff_datetime': str
            - 'passenger_count': int
            - 'pickup_loc_name': str
            - 'dropoff_loc_name': str
            - 'trip_distance': float
            - 'fare_amount': float
            Returns an empty DataFrame with these columns if any of the
            required CSV files ('trips.csv', 'drivers.csv', 'locations.csv')
            are missing.

        Notes
        -----
        The function reads the 'trips.csv', 'drivers.csv', and 'locations.csv'
        files from the database directory. It then merges these DataFrames and
        formats the resulting DataFrame according to the specified column types
        and names.
        """
        trip_path = os.path.join(self.data_dir, "trips.csv")
        driver_path = os.path.join(self.data_dir, "drivers.csv")
        location_path = os.path.join(self.data_dir, "locations.csv")
        columns = [
            "driver_lastname",
            "driver_givenname",
            "pickup_datetime",
            "dropoff_datetime",
            "passenger_count",
            "pickup_loc_name",
            "dropoff_loc_name",
            "trip_distance",
            "fare_amount",
        ]

        if (
            not os.path.isfile(trip_path)
            or not os.path.isfile(driver_path)
            or not os.path.isfile(location_path)
        ):
            return pd.DataFrame(columns=columns)

        else:
            trips = pd.read_csv(trip_path)
            drivers = pd.read_csv(driver_path)
            locations_pu = pd.read_csv(location_path)
            locations_do = pd.read_csv(location_path)
            locations_pu.rename(
                columns={"location_id": "pickup_loc_id"}, inplace=True)
            locations_do.rename(
                columns={"location_id": "dropoff_loc_id"}, inplace=True)

            df = pd.merge(trips, drivers, on="driver_id")
            df = pd.merge(df, locations_pu, on="pickup_loc_id").rename(
                columns={"loc_name": "pickup_loc_name"}
            )
            df = pd.merge(df, locations_do, on="dropoff_loc_id").rename(
                columns={"loc_name": "dropoff_loc_name"}
            )

            df.rename(
                columns={
                    "last_name": "driver_lastname",
                    "given_name": "driver_givenname",
                },
                inplace=True,
            )

            df.sort_values("trip_id", inplace=True)
            df = df.astype(
                {
                    "driver_lastname": "str",
                    "driver_givenname": "str",
                    "pickup_datetime": "str",
                    "dropoff_datetime": "str",
                    "passenger_count": "int",
                    "pickup_loc_name": "str",
                    "dropoff_loc_name": "str",
                    "trip_distance": "float",
                    "fare_amount": "float",
                }
            )

            df["driver_lastname"] = df["driver_lastname"].str.title()
            df["driver_givenname"] = df["driver_givenname"].str.title()
            return df[columns].reset_index(drop=True)

    def generate_statistics(self, stat):
        """Generate statistics based on the given 'stat' parameter.

        Parameters
        ----------
        stat : str
            The type of statistics to generate. Options are 'trip' for trip
            statistics, 'passenger' for passenger statistics 'driver' for
            driver statistics and 'all' for all statistics.

        Returns
        -------
        dict
            A dictionary or nested dictionary containing the generated
            statistics.

        Raises
        ------
        SakayDBError
            If the 'stat' parameter is not 'trip', 'passenger', 'driver' or
            'all'.
        """

        def load_trip_data():
            """Load trip data from the 'trips.csv' file and preprocess it.

            Returns
            -------
            tuple
                A tuple containing:
                - bool: True if the 'trips.csv' file exists and is successfully
                  loaded, False otherwise.
                - DataFrame or dict: A pandas DataFrame containing the trip
                  data if the file exists, otherwise an empty dictionary.
            """
            trip_file_path = os.path.join(self.data_dir, "trips.csv")
            if os.path.exists(trip_file_path):
                df_trips = pd.read_csv(trip_file_path)
                df_trips["pickup_datetime"] = pd.to_datetime(
                    df_trips["pickup_datetime"], format="%H:%M:%S,%d-%m-%Y"
                ).dt.floor("D")
                df_trips["day_name"] = (df_trips["pickup_datetime"]
                                        .dt.day_name())
                return True, df_trips
            else:
                return False, {}

        def trip_stat(is_valid, df_trips):
            """Generate trip statistics based on the given DataFrame.

            Parameters
            ----------
            is_valid : bool
                Whether the DataFrame is valid.
            df_trips : DataFrame
                A pandas DataFrame containing the trip data.

            Returns
            -------
            dict
                A dictionary where keys are day names and values are
                the average number of trips for each day.Returns an empty
                dictionary if the DataFrame is not valid.
            """
            if is_valid:
                trip_count = (
                    df_trips.groupby(["day_name", "pickup_datetime"])[
                        "trip_id"]
                    .count()
                    .reset_index()
                )
                average_trip = trip_count.groupby("day_name")["trip_id"].mean()
                return average_trip.to_dict()
            else:
                return {}

        def passenger_stat(is_valid, df_trips):
            """Generate passenger statistics based on the given DataFrame.

            Parameters
            ----------
            is_valid : bool
                Whether the DataFrame is valid.
            df_trips : DataFrame
                A pandas DataFrame containing the trip data.

            Returns
            -------
            dict
                A nested dictionary where the outer keys are the number of
                passengers and the inner keys are day names.The inner values
                are the average number of trips for each day and passenger
                count. Returns an empty dictionary if the DataFrame
                is not valid.

            """
            if is_valid:
                passenger_count = (
                    df_trips.groupby(
                        ["passenger_count", "day_name", "pickup_datetime"]
                    )["trip_id"]
                    .count()
                    .reset_index()
                )
                ave_passenger_count = (
                    passenger_count.groupby(
                        ["passenger_count", "day_name"])["trip_id"]
                    .mean()
                    .reset_index()
                )
                result_dict = (
                    ave_passenger_count.groupby("passenger_count")
                    .apply(lambda row: dict(zip(row["day_name"],
                                                row["trip_id"])))
                    .to_dict()
                )
                return result_dict
            else:
                return {}

        def driver_stat(is_valid, df_trips):
            """Generate driver statistics based on the given DataFrame.

            Parameters
            ----------
            is_valid : bool
                Whether the DataFrame is valid.
            df_trips : DataFrame
                A pandas DataFrame containing the trip data.

            Returns
            -------
            dict
                A nested dictionary where the outer keys are the full names of
                the drivers and the inner keys are day names.The inner values
                are the average number of trips for each day and driver.
                Returns an empty dictionary if the DataFrame is not valid or if
                the 'drivers.csv' file doesn't exist.
            """
            driver_file_path = os.path.join(self.data_dir, "drivers.csv")
            if os.path.exists(driver_file_path) and is_valid:
                drivers_dataframe = pd.read_csv(driver_file_path)
                merged_df = df_trips.merge(
                    drivers_dataframe, how="left", on="driver_id"
                )
                merged_df["full_name"] = (
                    merged_df["last_name"] + ", " + merged_df["given_name"]
                )
                driver_count = (
                    merged_df.groupby(["full_name",
                                       "day_name",
                                       "pickup_datetime"])[
                        "trip_id"
                    ]
                    .count()
                    .reset_index()
                )
                driver_count_avg = (
                    driver_count.groupby(["full_name", "day_name"])["trip_id"]
                    .mean()
                    .reset_index()
                )
                result_dict = (
                    driver_count_avg.groupby("full_name")
                    .apply(lambda row: dict(zip(row["day_name"],
                                                row["trip_id"])))
                    .to_dict()
                )
                return result_dict
            else:
                return {}

        is_valid_data, trip_data = load_trip_data()

        if stat == "trip":
            return trip_stat(is_valid_data, trip_data)

        elif stat == "passenger":
            return passenger_stat(is_valid_data, trip_data)

        elif stat == "driver":
            return driver_stat(is_valid_data, trip_data)

        elif stat == "all":
            return {
                "trip": trip_stat(is_valid_data, trip_data),
                "passenger": passenger_stat(is_valid_data, trip_data),
                "driver": driver_stat(is_valid_data, trip_data),
            }
        else:
            raise SakayDBError

    def plot_statistics(self, stat):
        """Generate and display plots for various types of statistics.

        Parameters
        ----------
        stat : str
            The type of statistic to plot. Options are 'trip', 'passenger',
            and 'driver'.

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot or matplotlib.figure.Figure
            The plot object generated by Matplotlib.

        Raises
        ------
        SakayDBError
            If the required CSV files do not exist or if an invalid `stat`
            argument is provided.
        """
        if stat == "trip":
            # Define file path and check existence
            trip_path = os.path.join(self.data_dir, "trips.csv")
            if not os.path.exists(trip_path):
                raise SakayDBError

            order_day = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]

            # Plotting
            ax = (
                pd.Series(self.generate_statistics(stat),
                          name="DateValue").reindex(
                    order_day
                )
            ).plot(kind="bar", figsize=(12, 8))
            ax.set(
                xlabel="Day of week",
                ylabel="Ave Trips",
                title="Average trips per day"
            )

            return ax

        elif stat == "passenger":
            # Define file path and check existence
            trip_path = os.path.join(self.data_dir, "trips.csv")
            if not os.path.exists(trip_path):
                raise SakayDBError

            # Prepare data for DataFrame
            passenger_count, day_names = zip(
                *[
                    (p, pd.DataFrame.from_dict(d, orient="index"))
                    for p, d in self.generate_statistics(stat).items()
                ]
            )

            # Create and format DataFrame
            df = pd.concat(day_names, keys=passenger_count).reset_index()
            df.rename(
                columns={
                    "level_1": "day_name",
                    "level_0": "passenger_count",
                    0: "ave_trips",
                },
                inplace=True,
            )
            df = df.pivot(
                index="day_name", columns="passenger_count", values="ave_trips"
            )

            # Reorder DataFrame
            order_day = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            df = df.reindex(order_day)

            # Plotting
            ax = df.plot(kind="line", figsize=(12, 8), marker="o")
            ax.set(xlabel="Day of week", ylabel="Ave Trips")

            return ax

        elif stat == "driver":
            # Define file paths and check existence
            trip_path = os.path.join(self.data_dir, "trips.csv")
            driver_path = os.path.join(self.data_dir, "drivers.csv")
            if not (os.path.exists(trip_path) and os.path.exists(driver_path)):
                raise SakayDBError

            # Prepare data for DataFrame
            drivers, day_names = zip(
                *[
                    (d, pd.DataFrame.from_dict(day, orient="index"))
                    for d, day in self.generate_statistics(stat).items()
                ]
            )

            # Create and format DataFrame
            df = pd.concat(day_names, keys=drivers).reset_index()
            df.rename(
                columns={
                    "level_1": "day_name",
                    "level_0": "driver_name",
                    0: "ave_trips",
                },
                inplace=True,
            )
            df = df.pivot(index="driver_name",
                          columns="day_name", values="ave_trips")

            # Define day order
            order_day = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]

            # Plotting
            fig, ax = plt.subplots(nrows=7, sharex=True, figsize=(8, 25))
            for i, day in enumerate(order_day):
                (
                    (
                        df[day]
                        .nlargest(5)
                        .reset_index()
                        .sort_values([day, "driver_name"],
                                     ascending=[True, False])
                    ).plot(ax=ax[i], kind="barh",
                           y=day, x="driver_name", legend=True)
                )
                ax[i].set(ylabel=None, xlabel="Ave Trips")

            return fig

        else:
            raise SakayDBError

    def generate_odmatrix(self, date_range=None):
        """Generate an Origin-Destination (OD) matrix based on trip data.

        Parameters
        ----------
        date_range : tuple of str, optional
            A tuple containing the start and end dates for filtering the
            trip data. Dates should be in the format '%H:%M:%S,%d-%m-%Y'.
            If not provided, no date filtering is applied.

        Returns
        -------
        pandas.DataFrame
            The generated OD matrix. Rows represent drop-off locations, and
            columns represent pick-up locations. The values in the matrix
            represent the  average number of trips between each pair of
            locations.

        Raises
        ------
        SakayDBError
            If the date_range tuple length is not 2, or if date conversion
            fails, or if the required CSV files do not exist.
        """

        # Define file paths
        trip_path = os.path.join(self.data_dir, "trips.csv")
        loc_path = os.path.join(self.data_dir, "locations.csv")

        # Check if both files exist, and read them into DataFrames if they do
        if all(os.path.exists(path) for path in [trip_path, loc_path]):
            df_trips = pd.read_csv(trip_path)
            df_locs = pd.read_csv(loc_path)
        else:
            return pd.DataFrame()

        df_trips = df_trips.merge(
            df_locs, left_on="pickup_loc_id",
            right_on="location_id", how="left"
        ).merge(df_locs, left_on="dropoff_loc_id",
                right_on="location_id", how="left")
        df_trips.rename(
            columns={"loc_name_x": "pickup_loc_name",
                     "loc_name_y": "dropoff_loc_name"},
            inplace=True,
        )
        df_trips = df_trips[["pickup_datetime",
                             "pickup_loc_name", "dropoff_loc_name"]]
        df_trips["pickup_datetime"] = pd.to_datetime(
            df_trips["pickup_datetime"], format="%H:%M:%S,%d-%m-%Y"
        )

        if isinstance(date_range, tuple):
            if len(date_range) != 2:
                raise SakayDBError

            try:
                start, end = date_range
                datetime_format = "%H:%M:%S,%d-%m-%Y"

                if start is not None:
                    start = pd.to_datetime(start, format=datetime_format)
                    df_trips = df_trips[df_trips["pickup_datetime"] >= start]

                if end is not None:
                    end = pd.to_datetime(end, format=datetime_format)
                    df_trips = df_trips[df_trips["pickup_datetime"] <= end]

                df_trips = df_trips.sort_values("pickup_datetime")

            except Exception:
                raise SakayDBError

        elif date_range is None:
            pass
        else:
            raise SakayDBError

        df_trips["pickup_datetime"] = df_trips["pickup_datetime"].dt.date
        df_trips["uniquedate"] = df_trips.groupby(
            ["pickup_loc_name", "dropoff_loc_name"]
        )["pickup_datetime"].transform("nunique")
        df_date_unique = df_trips.pivot_table(
            index="dropoff_loc_name",
            columns="pickup_loc_name",
            values="uniquedate",
            aggfunc="mean",
        )
        od_matrix = pd.crosstab(
            index=df_trips["dropoff_loc_name"],
            columns=df_trips["pickup_loc_name"]
        )
        od_matrix = od_matrix / df_date_unique
        od_matrix = od_matrix.fillna(0)

        return od_matrix
