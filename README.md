# EVStationDemand
Personal project mapping Electric Vehicle (EV) Stations in Vancouver against demand based on intersection traffic counts and parked cars

# Data

Data courtesy of the City of Vancouver Open Data Portal:

- [EV Charging Stations](https://opendata.vancouver.ca/explore/dataset/electric-vehicle-charging-stations/information/)
- [Intersection Traffic Movement](https://opendata.vancouver.ca/explore/dataset/intersection-traffic-movement-counts/information/)
- [Directional Traffic Counts](https://opendata.vancouver.ca/explore/dataset/directional-traffic-count-locations/information/)
- [Orthophoto Imagery](https://opendata.vancouver.ca/explore/dataset/orthophoto-imagery-2022/information/)
- [CARPK Dataset](https://paperswithcode.com/dataset/carpk)
- [VEDAI Dataset](https://downloads.greyc.fr/vedai/)
  
# Steps

1. Use the CARPK/VEDAI datasets and a suitable pretrained image recognition model (i.e. COCO), and fine tune it to count the number of cars present in a satelite image
2. Use City of Vancouver Orthophotography and this fine tuned network to create a density map of parked cars via Kriging
3. Overlay this layer with a mapping of EV Vehicle stations
4. Develop an algorithm to identify high parking density regions with low proximity to charging stations
5. (Optional) Use intersectional traffic data to supplement car density modelling, or to capture areas of high 'turnover rate' (areas with high parking and high mobile traffic, as opposed to high parking and low mobile traffic) to identify areas of high/ low charging session duration. 
