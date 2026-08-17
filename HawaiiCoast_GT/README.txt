--------------------------------------------------------------------------------
INTRODUCTION
--------------------------------------------------------------------------------
Title: Curated AIS for Hawaii's coast correlated with ground truth incidents.
Short title: HawaiiCoast_GT
Institution: Sandia National Laboratories
Author/POC: Amelia Henriksen

Funding Statement:
Sandia National Laboratories is a multimission laboratory managed and operated by
National Technology & Engineering Solutions of Sandia, LLC, a wholly owned
subsidiary of Honeywell International Inc., for the U.S. Department of Energy’s
National Nuclear Security Administration under contract DE-NA0003525. The views
expressed in the article do not necessarily represent the view of the U.S. DOE
or the United States Government.

Dataset Overview:
--------------------------------------------------------------------------------
Because of the high-risk nature of emergencies and illegal activities at sea, it
is critical that algorithms designed to detect anomalies from maritime traffic
data be robust. However, there exist no publicly available maritime traffic
datasets with real-world labelled anomalies. As a result, most anomaly detection
algorithms for maritime traffic are validated without ground truth. We introduce the
HawaiiCoast_GT dataset, the first ever publicly available automatic
identification system dataset with a large corresponding set of true anomalous
incidents. This dataset—cleaned and curated from raw Bureau of Ocean Energy
Management (BOEM) and National Oceanic and Atmospheric Administration (NOAA)
automatic identification system (AIS) data--covers Hawaii’s coastal waters for
four years (2017-2020) and contains 88,749,176 AIS points for a total of 2,622
unique vessels. 208 tracks are labelled corresponding to 154 labelled real world
incidents. The codebase used to curate the original AIS data is being made openly
available on GitHub.

Organization:
--------------------------------------------------------------------------------
This dataset consists of the following files:
  Documentation (main directory):
    - README.txt: The incredibly awesome and helpful file you are reading right
        now.
    - helper_functions.py: A set of functions to help with visualizing the
        HawaiiCoast_GT dataset and, in particular, the included incidents. Note
        that actual code used to **generate** the dataset is available on Github.
    - loading_and_visualization.ipynb: A jupyter notebook the demonstrates how
        to load and visualize HawaiiCoast_GT data.
    - vessel_type_codes_2018.csv: Derived from the United States Coast Guard 2018
        guidelines, available at https://coast.noaa.gov/data/marinecadastre/ais/VesselTypeCodes2018.pdf
        This csv maps the type code information from the USCG to the generalized
        vessel classes used in the dataset.
  AIS data: (AIS_data directory)
    - vessel_names_and_classes.csv: A primary list of the unique vessels in the
        dataset by MMSI, with derived names and classes.
    - Hawaii_<year>_<month>.csv: Monthly AIS covering sea surrounding Hawaii's
        coastline for year in [2017, 2018, 2019, 2020] and months
        [01, 02, 03, ... 12].
  Incident data: (incident_data directory)
    - hawaii_primary_trajectories_of_interest_2017_2020.csv: Contains the details
        for each of the incidents included in the labelled AIS dataset.
    - Hawaii_incident_source_key.bib: Contains the bibliographic details for all
        the records used to compile hawaii_primary_trajectories_of_interest_2017_2020.csv.


--------------------------------------------------------------------------------
AIS DATA
--------------------------------------------------------------------------------
This dataset was derived from MarineCadastre.gov's North American AIS data, which
is available for download by day.

Filename Format:
  1. Hawaii_<year>_<month>.csv
      - year: [2017, 2018, 2019, 2020]
      - month: [01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12]
  2. vessel_names_and_classes.csv

File Format:
--------------------------------------------------------------------------------
1. Hawaii_<year>_<month>.csv
  - Columns:
      1.	MMSI: Nine digit unique vessel identifier.
      2.	datetime_utc: Date/time value in Coordinated Universal Time (UTC).
            Corresponds to BaseDateTime in the original MarineCadastre data.
            Format <year>_<month>_<day>T<hour>:<minute>:<second>
      3.	lat: Latitudinal position in decimal degrees. Corresponds to 'LAT' in
            the original MarineCadastre data.
      4.	lon: Longitudinal position in decimal degrees. Corresponds to 'LON' in
            the original MarineCadastre data.
      5.	speed_over_ground_knots: Speed over ground in knots. Maximum possible
            value 120.3. Corresponds to 'SOG' in the original MarineCadastre data.
      6.	course_over_ground_deg: Course over ground in degrees (0 to 360).
            Corresponds to 'COG' in the original MarineCadastre data.
      7.	heading_deg: Heading in degrees. Corresponds to 'Heading' in the
            original MarineCadastre data. Note a heading of "511" indicates that
            the heading is not available.
      8.	vessel_name: Vessel's name, note the name is always in capital
            letters. Methods for this column are described in Methods step 3
            below.
      9.	IMO: International maritime organization number, Format IMO<#>.
            Corresponds to 'IMO' in the original MarineCadastre data.
      10.	call_sign: Corresponds to 'CallSign' in the original MarineCadastre
            data.
      11.	vessel_type_code: Integer vessel type code corresponding to
            MarineCadastre/The United States Coast Guard 2018 guidelines,
            available at https://coast.noaa.gov/data/marinecadastre/ais/VesselTypeCodes2018.pdf
      12.	status: Navigation status defined by the Convention on the
            International Regulations for Preventing Collisions at Sea, 1972
            (COLREGs). Corresponds to 'Status' in the original MarineCadastre
            data.
      13.	length_m: Length of the vessel in meters. Corresponds to 'Length' in
            the original MarineCadastre data.
      14.	width_m: Width of the vessel in meters. Corresponds to 'Width' in the
            original MarineCadastre data.
      15.	draft_depth_m: Draft depth of a vessel in meters. Corresponds to
            'Draft' in the original MarineCadastre data.
      16.	cargo_type_code: Cargo type code as defined by the NAIS specification
            and codes. Corresponds to 'Cargo' in the original MarineCadastre data.
      17.	Transceiver_class: Class of AIS transceiver (A or B). Corresponds to
            ‘TransceiverClass’ in the original MarineCadastre data.
      18.	datetime_hst: datetime_utc converted to Hawaii Standard Time (hst).
      19.	vessel_class: Class label derived as described in Methods step 4 below.
      20. distances_km: The distance between subsequent points with the same MMSI
            value in kilometers. Details in step 5.2 of the Methods section
            below.
      21. comput_speed_knots: The computed speed between two subquent points
            with the same MMSI value in knots. Details in step 5.3 of the Methods
            section below.

2. vessel_names_and_classes.csv
  - Columns:
    1. MMSI: Each unique mmsi over all AIS points in the dataset.
    2. vessel_name: A list of the verified and standardized vessel names
      affiliated with the mmsi in the dataset.
    3. vessel_class: The vessel class determined for each unique mmsi in the data
      set as descibed in methods step 4 below.

Methods:
--------------------------------------------------------------------------------
Note that the code used to execute the following steps is being made available
on GitHub.

  1. Initial Data Gathering: For each day between January 1, 2017 and December
    31, 2020:
      1.1 Download the AIS data for the entire North American region from
      https://coast.noaa.gov/htdata/CMSP/AISDataHandler/<year>/index.html
      1.2 Downsample to the following bounding box, capturing Hawaii's coastline:
          - Longitude range: -161.70558, -152.98331
          - Latitude range: 24.07175, 18.13869
      1.3 Compile daily downsampled data into csv files for each month.
      1.4 Create a new column of data converting the datetime_utc to Hawaii
            standard time. This is important for correlating anomalous event
            reports (which are typically in local time) to the AIS data.

  2. Initial Data Cleaning: For each month in 2017, 2018, 2019, and 2020:
      2.1 Group data by indexing by MMSI
      2.2 Remove any points for which there is only one AIS point associated with
        the given MMSI in that monthly dataset.
      2.3 Remove any entries for which the MMSI is not 9 characters long
        (invalid).

  3. Name Discovery and cleanup:
      3.1 Compile a list of unique MMSI values for the entire dataset, and all
        names associated with each MMSI value over the course of the dataset.
      3.2 If an MMSI with a 'nan' in the 'vessel_name' column has a unique name
        elsewhere in the dataset, set the 'nan' value to the unique name.
      3.3 If an MMSI has multiple names that are variations on the same (e.g.
        'MARIE H' and 'MARIE H.') choose one and label all 'vessel_name' values
        for that MMSI accordingly.
      3.4 For this dataset, 3 mmsi values had two different names that were not
        variations of one another. There is evidence that this is because these
        three vessels' names were changed during the four year period from 2017
        to 2018. For these mmsi values, we leave the disparate names in place,
        to appropriately capture the name change in the data.
          - 255805759: 'MARIA P', 'SAMOANA'
          - 338189586: 'SD-1046', 'SD-1025'
          - 366230000: 'SLNC MAGOTHY', 'NORFOLK'
      3.5 For the remaining unnamed data points, we fill the 'vessel_name' values
        by web scraping for the names based on the associated mmsi values.
          3.5.1 First try scraping www.marinetraffic.com for the vessel name.
          3.5.2 For the remaining missing names, try a second pass by searching
            marinevesseltraffic.com. Note that the name format for
            marinevesseltraffic.com is not consistent with AIS data, so these
            names had to be edited before addition to the dataset.
          3.5.3 For the still remaining missing names, try a third pass by
            searching www.myshiptracking.com.
      3.6 Use the finalized map of MMSI values to vessel name to fill in or
        standardize the vessel classes for all AIS points.
      3.7 Save this data in vessel_names_and_class.csv.

      IMPORTANT NOTE:
        For this dataset, we were able to reduce the number of unnamed MMSI values
        from 551 to 50. HOWEVER, this web scraping method has a small chance of
        inaccuracy, because it is difficult to ascertain whether a name change
        took place in 2017-2020. Our web scraping captured the names as of March,
        2023 for the unnamed 551 MMSI values. These represent less than a fifth of
        the unique vessels in the dataset, and most vessels are unlikely to have
        changed their name. However, in the case that the modern name does not
        match the missing name from the MMSI during the specified time period, it
        simply means we would be less likely to find an incident associated with
        that name for our case study incident set.

  4. Vessel Class discovery and cleanup
      4.1 Create a new data column for all monthly datasets, 'vessel_class'
      4.2 Use the list of unique MMSI values for the entire dataset, and
        compile all 'vessel_type_code' values associated with each MMSI value over
        the course of the dataset.
      4.3 Load 'vessel_type_codes_2018.csv' to create a map between MarineCadastre's
        'vessel_type_2018' and the appropriate value in 'ship_classes_for_classification'
        This will map 'vessel_type_code' values to 'vessel_class' values.
      4.4 Use this map to and the 'vessel_type_code' values for each MMSI value to
        obtain a single 'vessel_class' value for each MMSI value.
          4.4.1 For MMSI values with a unique, non-nan 'vessel_type_code', the
            'vessel_class' value is simply set accordingly.
          4.4.1 For MMSI values with multiple 'vessel_type_code' values in the
            dataset, remove any 'nan' values. If one unique 'vessel_type_code' code
            remains, set the 'vessel_class' accordingly.
          4.4.2 For MMSI values with multiple 'vessel_type_code' values that map to
            the same 'vessel_class', set the 'vessel_class' accordingly.
          4.4.3 For MMSI values with multiple 'vessel_type_code' values where one
            maps to 'Other' and the other maps to a more specific identifier,
            set the 'vessel_class' using the more specific identifier.
              - Note, in this dataset the only exception to this rule was MMSI
                431999529, which had 'Other' and 'Not available' as the
                classes in the dataset. We set this value to 'Other.'
          4.4.4 For vessels that still have multiple possible vessel classes,
            use an internet search to determine the best fit. For this dataset,
            this was possible manually as only 4 MMSI's had distinctly
            different classes listed in the dataset:
              - 367507280: 'Fishing', 'Pleasure Craft/Sailing' -> 'Fishing'
              - 367556290: 'Fishing', 'Pleasure Craft/Sailing' -> 'Fishing'
              - 367798710: 'Fishing', 'Pleasure Craft/Sailing' -> 'Pleasure Craft/Sailing'
              - 368070780: 'Fishing', 'Pleasure Craft/Sailing' -> 'Fishing'
            Note that our codebase prompts the user in these cases and lets them
            choose the appropriate class.
      4.5 For the remaining MMSI values without 'vessel_class' information, we
        fill the 'vessel_class' values by web scraping.
            4.5.1 First try scraping www.marinevesseltraffictraffic.com for the
              vessel type. Note that there are some discrepancies in class name
              convention, for example, marinevesseltraffic uses 'Miltary OPS'
              rather than 'Military' and 'Tug' instead of 'Tug Tow'. Map names
              appropriately.
            4.5.2 For the remaining missing 'vessel_class' values, try a second
              pass by searching marinetraffic.com. Again, map between the naming
              conventions appropriately.
      4.6 Use the finalized map of MMSI values to vessel class to fill in
        the vessel classes for all AIS points.
      4.7 Save this data in vessel_names_and_class.csv.

      IMPORTANT NOTE: We used the techniques listed above to fill in as much of
        the 'vessel_class' information as possible throughout the dataset. Using
        our web scraping technique, we were able to reduce the vessels with
        missing classes from 612 to 338. This provides useful information for
        classification problems on our dataset. However, we left the reported
        'vessel_type_code' values unchanged, since the actual reported code or the
        fact that 'vessel_type_code' was unreported or potentially concealed during a
        given trajectory may be of interest.

  5. Vessel computed distance and speed:
      6.1 For each month, for each unique mmsi within the month, compile all
          sequential points in time as a trajectory using Sandia National Lab's
          tracktable application.
      6.2 Use the built in tracktable distance functions to compute the haversine
          distances between two sequential 'lat'/'lon' points. Store this in a
          new column, 'distances_km.'
      6.3 Use the 'datetime_utc' and these computed distance values to approximate
          the speed of the vessel in knots at each point. Speeds are assigned to
          the second of each pair of two points, so the first point of each
          vessel 'trajectory' has no associated speed value. Store this in a new
          column: 'comput_speed_knots'

      IMPORTANT NOTES:
        NOTE 1: Note that in step 5.1, we do not split the trajectory by
          time gap or lack of movement. All points with the same MMSI are treated
          as a single trajectory each month. Thus for data users building
          tracks using splits such as low convex hull, time gap between points,
          etc, the speeds at the beginning of split trajectories should be
          disregarded.
        NOTE 2: In step 5.2, we note that the haversine distance is an
          approximation to the distance between two points on the surface of the
          earth. For sufficiently distant points on the globe, this distance
          function has a small amount of error. However, for this coastal dataset
          the spatial bounding box is small enough that any distance error from
          treating the earth as a sphere rather than an ellipsoid is negligible.

  6. Incident labels:
    6.1 For each vessel in Hawaii_primary_trajectories_of_interest_2017_2020.csv
      label the AIS points between ais_incident_start_bound_hst and
      ais_incident_end_bound_hst with the corresponding incident_num value.


--------------------------------------------------------------------------------
INCIDENT DATA
--------------------------------------------------------------------------------
Files:
  1. Hawaii_primary_trajectories_of_interest_2017_2020.csv
  2. Hawaii_incident_source_key.bib

File Format:
--------------------------------------------------------------------------------
1. Hawaii_primary_trajectories_of_interest_2017_2020.csv
  Columns:
    1. incident_num: simple index used to correlate AIS points to the incident.
        Multiple vessels involved in the same incident all have the same
        incident_num.
    2. MMSI: The maritime mobile service identity of the involved vessel (9 digit
        unique identifier)
    3. vessel_name: The name (as included in the AIS data) of the involved vessel.
    4. vessel_class_from_ais: The class (as included in the AIS data) of the
      involved vessel.
    5. source: source category (or list of categories for multiple references) for
        the records used to find the incident.
        Options:
          1. USCG Maritime Information Exchange Incident Investigation Reports ('IIR')
          2. USCG National Response Center
          3. Newspaper News
          4. Defense Visual Information Distribution Service
          5. USCG Archived List of IMO reportable detentions
          6. USCG Annual Deficiency Report
    6. reference_key: Unique key(s) corresponding to the exact reference report(s)
        for the incident. These reference are stored as bibtex entries in
        haw_source_key.bib.
    7. report_based_incident_description: A description of the incident based
        specifically on the record contents. For USCG list of IMO reportable
        detentions and USCG Annual Deficiency Report this includes USCG specific
        codes.
    8. report_start_date_of_interest_hst: The start date of the incident in Hawaii
        Standard Time (HST) according to the reports. Note that this may be
        different from the start time of available AIS in the dataset.
    9. report_end_date_of_interest_hst: The end date of the incident in Hawaii
        Standard Time (HST) according to the report. Note that not all reports
        included an end date, so this column is left blank for many incidents.
    10: report_incident_time_of_interest_hst: The exact time (hour:min) of the
        incident on the report_start_date_of_interest_hst included in the report.
        Note that this is included in many but not all IIR reports and NRC
        reports.
    11. report_vessel_role: A summary phrase for the role of the vessel in the
        incident, according to the report.  This is mostly at play for USCG IIR
        reports, which have common summary phrases in the report title. Note that
        the report_vessel_role is distinct from the ais_incident_type, which
        indicates the type of anomaly actually present in the data vs. reported.
    12. report_vessel_class: The class or type of vessel as included in the
        report. Used to verify the AIS correlation.
    13. report_location: Where the incident took place according to the report(s).
        Each report type has different ways of referring to the location, this is
        extracted directly from each report without standardization. Note some
        reports had no location information, in which case this information is
        omitted.
    14. ais_incident_start_bound_hst: The datetime_hst value where the AIS
        incident label begins.
    15. ais_incident_end_bound_hst: The datetime_hst value where the AIS
        incident label ends.
    16. ais_incident_type: The type(s) of incident or abnormal behavior actually
        available in the AIS. This is distinct from the report_vessel_role,
        because an event included in the report may have happened outside the
        coastal bounding box or when the vessel was not broadcasting AIS. This
        can be thought of as an anomaly label for the AIS. Note that some vessels
        have multiple anomalous behaviors, for example a loss of propulsion and
        an irregular tow.
        Options:
          - Allision: Refers to a moving vessel hitting a stationary vessel.
          - Boarding: Refers to law enforcement boarding the vessel.
          - Collision: Refers to two moving vessels hitting one another.
          - Container loss: Refers to a vessel losing shipping containers.
          - Fire: Refers to fire aboard a ship.
          - Flooding: Refers to water breaching the hull of a vessel.
          - Grounding: Refers to a vessel hitting the ground/running aground.
          - Helper tow: Refers to a tow vessel that performs a tow, but in
              response to a ground truth incident (essentially the tow
              performing towing duties is normal, but is still involved in the
              incident.)
          - Helper vessel: Refers to a vessel that generally provides aid in an
              incident other than performing a tow, which has a distinct label
              ("Helper tow" or "Irregular tow")
          - Injury/medical emergency: Refers to a passenger or crewmember
              being injured or having a medical emergency of some kind. In this
              case, death is labelled as a medical emergency rather than as a
              separate label.
          - Irregular tow: Refers to a non-tow vessel towing another vessel, or
              a vessel that is typically not towed (not a barge, for example)
              being towed.
          - Loss of power: Refers to a vessel losing (typically electrical)
              power.
          - Loss of steering/maneuverability: Refers to a vessel losing steering
              or maneuverability (often happens in conjunction with other
              incident types).
          - Loss/reduction of propulsion: Refers to a vessel losing (or reducing)
              it's ability to move forward.
          - Material failure: Refers to a ship component breaking or otherwise
              not functioning properly.
          - Pollution: Refers to activities such as improper disposal of bilge
              waste, oil leaks, etc.
          - Property damage: Covers all incidents where a vessel is damaged
              in ways other than specifically labelled (e.g. Allision, Collision,
              Material failure, Grounding, or Sinking).
          - Route deviation: Refers to a vessel making an incident specific trip
              or route change (e.g. a vessel changing course to help, a vessel
              encountering a failure and turning around to return to port rather
              than continuing).
          - Sinking: Refers to a vessel sinking (non-submersible vessel
              submerging)
    17. ais_incident_evidence: A summary of how much obvious evidence is available
        in the corresponding AIS trajectory visualization and speed profile.
        Options:
          1. High evidence: The ais_incident_type is exhibited clearly when the
              AIS lat/lon points from ais_incident_label_start bound to
              ais_incident_label_end_bound_hst are visualized, or the speeds
              between these time bounds are visualized.
          2. Good evidence: The visualization and speed profile seem to indicate
              the event, but there may not be enough detail in the report to
              100% indicate the exact ais_incident_type.
          3. Some evidence: There is some kind of evidence that the incident
              occurred, but most is lacking.
          4. No obvious evidence: There is no obvious evidence that the event
              occurred, but AIS data is still available that corresponds to the
              incident vessel at the report-indicated-time. Incidents with no
              OBVIOUS evidence are a rich area to develop more specific features
              that may reveal the ground truth.
    18. ais_availability_description: Expands on the ais_incident_evidence summary
        label. Explains how the trajectories in the AIS data correspond (or do
        not correspond) to the report descriptions. This description is critical
        to understanding the ground truth in the dataset.

2. Hawaii_incident_source_key.bib: Standard bibtex file.

Methods:
--------------------------------------------------------------------------------
1. Report collection:
  1.1 USCG Maritime Information Exchange Incident Investigation Reports ('IIR')
    1.1.1 Access the US Coast Guard's publicly available Maritime Information
      Exchange Incident Investigation Report portal at https://cgmix.uscg.mil/IIR/IIRSearch.aspx.
    1.1.2 Set the search date range from 1/1/2017 to 12/31/2020.
    1.1.3 Use each vessel name in the list of compiled vessel names (from the AIS
      data processing step 3) as a general keyword search.
    1.1.4. By hand, examine the resulting articles for MISLE Originating unit
      sector honolulu and download them for further analysis.

    IMPORTANT NOTE: At the time of this dataset's creation, IIR reports had to
      be downloaded individually by hand because there is not a web API available
      to interface with the IIR search engine, and the authors lack the javascript
      ability to scrape this data's difficult interface. A very important area
      of future work will be to update the codebase for this dataset's generation
      with a way to automatically download the IIR reports.

      We also note that for datasets over smaller stretches of time (such as a
      month) it is simpler to download all reports and filter by sector and
      vessel name from the html files. For a period of four years, however,
      it was significantly faster to narrow the number of records to review by
      searching first by vessel name.

      Hopefully the IIR interface will include tags for the MISLE Originating
      unit so that records can be viewed by sector, but that is not yet a
      feature of this record source.

  1.2 USCG National Response Center
    1.2.1. Download the 2017, 2018, 2019, and 2020 USCG National Response Center
      annual reports at https://nrc.uscg.mil/.
    1.2.2. Filter incidents with location state 'HI'.
    1.2.3. Filter incidents with incident type 'VESSEL'.
    1.2.4. Filter for incidents that include vessel name
      NOTE: Vessel name is the only form of identifier in the NRC reports, which
        is needed to correlate the ground truth to the AIS. Future work could
        note ship behavior around a location on interest without identifing the
        actual vessel of interest, but that is outside the scope of this dataset.
    1.2.5. Filter for the following incident types that have potential to be of
      interest:
        - 'UNKNOWN'
        - 'VESSEL SINKING'
        - 'EQUIPMENT FAILURE'
        - 'OTHER'
        - 'OPERATOR ERROR'
        - 'DUMPING'
        - 'NATURAL PHENOMENON'
        - 'TRESPASSER'
    1.2.6. Further review the remaining entries by hand by reading the description for
      relevance (e.g. a description corresponding to an incident that happened
      while a vessel was underway, rather than moored. Spilling 5 drops of paint
      in the water is not relevant).
    1.2.7 Filter for vessel names that appear in the dataset (using the compiled
      list of vessel names from step 3 of the AIS data cleanup methods).

    IMPORTANT NOTE: Because reviewing entry descriptions is time consuming, it
    can be significantly faster to reverse steps 1.2.6 and 1.2.7, filtering
    first for vessel names that appear in the dataset and then reviewing the
    descriptions. We listed the steps as we executed them for reproducibility,
    but note that because very few entries with vessel names in the AIS dataset
    were available it makes far more sense to filter for names first.

  1.3 Newspaper News
    1.3.1 Use google news to search for vessel incident related articles from
      2017 to 2020. Example keyword searches:
        - "Hawaii", "Sinking"
        - "Hawaii", "Boat", "Accident"
        - "Hawaii", "US Coast Guard"
        - "Hawaii", "Illegal fishing"
    1.3.2 Use similar keyword searches in Hawaii-specific or maritime specific
      journals compiled from these google news searches.
    1.3.3. Check to see if any vessel names are included in the article.
    1.3.4. Filter for vessel nmaes that appear in the dataset using the compiled
      list of vessel names, as with other records.

  1.4 Defense Visual Information Distribution Service:
    1.4.1. Access the Defense Visual Information Distribution Service (DVIDs)
      news search portal: https://www.dvidshub.net/search?filter[type]=news
    1.4.2. Set the search dates from 2017 to 2020.
    1.4.3. Aggregate news articles using search phrases such as 'Hawaii' and
      'Sector Honolulu.'
    1.4.4. List any vessel names included in the reports, and search for those
      names in the compiled list of vessels from the AIS dataset.

  1.5 USCG Archived List of IMO reportable detentions
    1.5.1. Download the Archived 2017, 2018, 2019, and 2020 lists of IMO reportable
        detentions provided by the US Coast Guard Port State Control Division.
        At the time of this dataset creation, these were available for years
        2015-2023 at
        https://www.dco.uscg.mil/Our-Organization/Assistant-Commandant-for-Prevention-Policy-CG-5P/Inspections-Compliance-CG-5PC-/Commercial-Vessel-Compliance/Foreign-Offshore-Compliance-Division/Port-State-Control/Detentions/
    1.5.2. Search for reports in sector Honolulu.
    1.5.3. Find the name of the involved vessel in the report and search for it
      in the compiled list of vessel names from step 3 of the AIS data cleanup
      method.

  1.6 USCG Annual Deficiency Report:
    1.6.1. Download the 2018-2020 US Coast Guard annual deficiency reports at
        https://www.dco.uscg.mil/Our-Organization/Assistant-Commandant-for-Prevention-Policy-CG-5P/Inspections-Compliance-CG-5PC-/Commercial-Vessel-Compliance/MISLE-DEF-DATA-REPORT/
        At the time this dataset was created, the 2018-2023 deficiency reports
        were available.
    1.6.2. Filter out only the deficiency reports issued in Sector Honolulu
      (which contains our region of interest).
    1.6.3. Filter for system codes that have potential connection to maritime
        trajectory anomalies:
          02: Structural Conditions
          03: Water/Weathertight Conditions
          06: Cargo Operations Including Equipment
          10: Safety of Navigation
          13: Propulsion and Auxiliary Machinery
          14: Pollution Prevention
    1.6.4. Filter for 'Resolution Action' values that prevent a vessel from
        travelling without resolving the deficiency issues.
          17: Rectify deficiencies prior to departure
          30: Ship detained
          60: Rectify deficiencies prior to movement
    1.6.5. Filter for vessel names that appear in the dataset (using the compiled
      list of vessel names from step 3 of the AIS data cleanup methods).

2. Initial AIS correlation check:
  2.1. For each report source, compile the following information:
    2.1.1 Name of vessel(s) involved in the incident
      -col 3: vessel name.
    2.1.2 MMSI associated with each vessel name (looking up using compiled
      dataset vessel information)
      -col 2: MMSI
    2.1.3 Any dates listed in the report in Hawaii Standard Time (HST)
      - col 8: report_start_date_of_interest_hst
      - col 9: report_end_date_of_interest_hst.
  2.2. For each MMSI and incident date, check that that MMSI appears in the AIS
    data for the listed incident month. If not, remove the report entry.
  2.3. For the remaining vessel entries and their corresponding reports, compile
    the following information:
      2.3.1 Any listed locations
        - col 13: report_location
      2.3.2 A brief description of the report contents/incident details:
        - col 7: report_based_incident_description
      2.3.3 A summary label for the role of the vessel involved. For the vessel
        that is experiencing the incident, this is often included in IIR titles.
        For other involved vessels, we typically include a label of "Helper
        vessel" for vessel role.
        - col 11: report_vessel_role
      2.3.4 The specific time of the incident, if included in the report, in
        Hawaii Standard Time (HST).
        - col 10: incident_time_of_interest_hst.
      2.3.5 The type/class of vessel as described in the report.
        - col 12: report_vessel_class.
      2.3.6. The overall source category of the report
        - col 5: source

3. Final incident check: For the remaining reports, do the following (by hand)
  3.1 Check the days of the month that the mmsi for the involved vessel(s) are
    available in the AIS. If the days of the month are not affiliated with the
    incident dates, remove the report.
  3.2 Visualize the included AIS points on/around the time of the incident to
    check if part or all of the incident trajectory is available in the AIS.
    Further visualize the speed profile associated with the incident, and
    mark any exact times associated with the incident on the speed profile
    (exact times of accidents or failures are often included in IIR reports and
    National Response Center reports).
  3.3 Use this visualization to fill in the following:
    3.3.1 Analyze the AIS evidence and how it aligns with the
      report based incident description. Detail any visual/speed profile
      evidence for the incident.
        - col 18: ais_availability_description
    3.3.2 Summarize the amount of evidence from the ais_availability_description
      as High evidence, Good evidence, Some evidence or No obvious evidence.
        - col 17: ais_incident_evidence
    3.3.3 Summarize the incident types covered by the AIS. For example, if a
      loss of propulsion event isn't available in the AIS but the subsequent
      anomalous tow is, this would have an AIS incident type of "irregular tow."
      Details for these categories are outlined in the column descriptions above.
        - col 16: ais_incident_type
    3.3 Datetimes that bound the actually available AIS points that contain
      related to the incident. This is highly dependent on the nature of the
      incident.
        - col 14: ais_incident_start_bound_hst
        - col 15: ais_incident_end_bound_hst

4. Finalize the incidents
  4.1. For each incident that is actually to be included in the dataset, create
    a bibtex entry with the relevant bibliographic reference information in
    Hawaii_incident_source_key.bib. For each incident vessel entry in Hawaii_primary_trajectories_of_interest_2017_2020.csv
    include the key for each reference.
    - col 6: reference_key.
  4.2. Sort the vessel entries by ais_incident_start_bound_hst and assign each
    incident a incident number (col 1: incident_num). Vessels that are part of
    the same incident have the same incident number.
