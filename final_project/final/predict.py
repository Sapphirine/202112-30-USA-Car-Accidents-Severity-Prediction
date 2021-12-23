import overpy
import datetime
import time
from tzwhere import tzwhere
from pytz import timezone
import pandas as pd
import requests
import json
from geopy.geocoders import Nominatim

import numpy as np

from sklearn.preprocessing import LabelEncoder
import joblib

SEVERITY_DICT = {
        1:"Level 1 (minor impact)",
        2:"Level 2 (medium impact)",
        3:"Level 3 (significant impact)",
        4:"Level 4 (high impact)"
    }

class Backend:

    def __init__(self):
        self.model = joblib.load('final/dt.pkl')
        self.geolocator = Nominatim(user_agent="geoapiExercises")
        self.map = overpy.Overpass()
        self.weather = "4657b71d1aacb5f317fb821a025f50a1"
        self.test = pd.DataFrame(columns=['Start_Lat', 'Start_Lng',
                                          'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
                                          'Wind_Direction', 'Wind_Speed(mph)', 'Weather_Condition',
                                          'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway',
                                          'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal',
                                          'Turning_Loop', 'Civil_Twilight', 'Year', 'Month', 'Weekday', 'Day', 'Hour',
                                          'Minute'], index=[0])

    # Time  'Year', 'Month', 'Weekday', 'Day', 'Hour', 'Minute'
    def get_time(self, location):
        tz = tzwhere.tzwhere()
        time_zone = tz.tzNameAt(location[0], location[1])
        timestamp = time.time()
        str_t = datetime.datetime.fromtimestamp(timestamp, timezone(time_zone)).strftime('%Y-%m-%d %H:%M:%S')
        t = pd.to_datetime(str_t)
        self.test['Year'] = t.year
        self.test['Month'] = t.month
        self.test['Weekday'] = t.weekday()
        self.test['Day'] = t.day
        self.test['Hour'] = t.hour
        self.test['Minute'] = t.minute
        self.test['Day'] = t.day
        if t.hour >= 12:
            self.test['Civil_Twilight'] = 'Night'
        else:
            self.test['Civil_Twilight'] = 'Day'

    # POI   'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station',
    #       'Stop', 'Traffic_Calming', 'Traffic_Signal',
    def get_POI(self, location):
        POI = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station',
               'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']
        for item in POI:
            self.test[item] = False
        min_lat = location[0] - 0.0005
        max_lat = location[0] + 0.0005
        min_lon = location[1] - 0.0005
        max_lon = location[1] + 0.0005
        query = "[out:json];node(%s,%s,%s,%s);out;" % (min_lat, min_lon, max_lat, max_lon)
        self.nodes = self.map.query(query)
        for node in self.nodes.nodes:
            if 'amenity' in node.tags:
                self.test['Amenity'] = True

            if ('bump' in node.tags):
                self.test['Bump'] = True

            if ('crossing' in node.tags):
                self.test['Crossing'] = True

            if 'give_way' in node.tags.values():
                self.test['Give_Way'] = True

            if ('junction' in node.tags):
                self.test['Junction'] = True

            if ('noexit' in node.tags):
                self.test['No_Exit'] = True

            if ('railway' in node.tags):
                self.test['Railway'] = True

            for value in node.tags.values():
                if 'bump' in value:
                    self.test['Bump'] = True
                if 'crossing' in value:
                    self.test['Crossing'] = True
                if 'junction' in value:
                    self.test['Junction'] = True
                if 'railway' in value:
                    self.test['Railway'] = True
                if 'roundabout' in value:
                    self.test['Roundabout'] = True
                if 'station' in value:
                    self.test['Station'] = True
                if 'stop' in value:
                    self.test['Stop'] = True

            if 'traffic_calming' in node.tags:
                self.test['Traffic_Calming'] = True

            if ('traffic_signals' in node.tags) or ('traffic_signals' in node.tags.values()):
                self.test['Traffic_Signal'] = True

    # Weather   'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Direction', 'Wind_Speed(mph)',
    #           'Precipitation(in)', 'Weather_Condition',
    def get_weather(self, location):
        url = "https://api.openweathermap.org/data/2.5/onecall?lat=%s&lon=%s&appid=%s&units=imperial" % \
              (location[0], location[1], self.weather)
        response = requests.get(url)
        data = json.loads(response.text)
        self.data = data
        self.test['Temperature(F)'] = data['current']['temp']
        self.test['Humidity(%)'] = data['current']['humidity']
        self.test['Pressure(in)'] = data['current']['pressure'] * 0.03
        self.test['Visibility(mi)'] = data['current']['visibility'] * 0.001
        degree = data['current']['wind_deg']
        dir = ["North", "NNE", "NE", "ENE", "East", "ESE", "SE", "SSE", "South", "SSW", "SW", "WSW", "West", "WNW",
               "NW", "NNW", "North"]
        self.test['Wind_Direction'] = dir[round(degree / 22.5)]
        self.test['Wind_Speed(mph)'] = data['current']['wind_speed']
        #    if 'rain_3h' in data['current']:
        #        self.test['Precipitation(in)'] = data['current']['rain_3h']
        #    elif 'rain_1h' in data['current']:
        #        self.test['Precipitation(in)'] = data['current']['rain_1h']
        #    else:
        #        self.test['Precipitation(in)'] = data['current']['rain_1h'] = 0
        self.test['Weather_Condition'] = data['current']['weather'][0]['main']

    def encoding(self):

        self.test['Wind_C'] = np.where(
            self.test.loc[0, 'Wind_Direction'].startswith('C'), 1, 0)
        self.test['Wind_E'] = np.where(
            self.test.loc[0, 'Wind_Direction'].startswith('C'), 1, 0)
        self.test['Wind_N'] = np.where(
            self.test.loc[0, 'Wind_Direction'].startswith('C'), 1, 0)
        self.test['Wind_S'] = np.where(
            self.test.loc[0, 'Wind_Direction'].startswith('C'), 1, 0)
        self.test['Wind_V'] = np.where(
            self.test.loc[0, 'Wind_Direction'].startswith('C'), 1, 0)
        self.test['Wind_W'] = np.where(
            self.test.loc[0, 'Wind_Direction'].startswith('C'), 1, 0)

        self.test.drop('Wind_Direction', axis=1, inplace=True)

        self.test['Weather_Fair'] = np.where(
            self.test['Weather_Condition'].str.contains('Fair', case=False, na=False), 1, 0)
        self.test['Weather_Cloudy'] = np.where(
            self.test['Weather_Condition'].str.contains('Cloudy', case=False, na=False), 1, 0)
        self.test['Weather_Clear'] = np.where(
            self.test['Weather_Condition'].str.contains('Clear', case=False, na=False), 1, 0)
        self.test['Weather_Overcast'] = np.where(
            self.test['Weather_Condition'].str.contains('Overcast', case=False, na=False), 1, 0)
        self.test['Weather_Snow'] = np.where(
            self.test['Weather_Condition'].str.contains('Snow|Wintry|Sleet', case=False, na=False), 1, 0)
        self.test['Weather_Haze'] = np.where(
            self.test['Weather_Condition'].str.contains('Smoke|Fog|Mist|Haze', case=False, na=False), 1, 0)
        self.test['Weather_Rain'] = np.where(
            self.test['Weather_Condition'].str.contains('Rain|Drizzle|Showers', case=False, na=False), 1, 0)
        self.test['Weather_Thunderstorm'] = np.where(
            self.test['Weather_Condition'].str.contains('Thunderstorms|T-Storm', case=False, na=False), 1, 0)
        self.test['Weather_Windy'] = np.where(
            self.test['Weather_Condition'].str.contains('Windy|Squalls', case=False, na=False), 1, 0)
        self.test['Weather_Hail'] = np.where(
            self.test['Weather_Condition'].str.contains('Hail|Ice Pellets', case=False, na=False), 1, 0)
        self.test['Weather_Thunder'] = np.where(
            self.test['Weather_Condition'].str.contains('Thunder', case=False, na=False), 1, 0)
        self.test['Weather_Dust'] = np.where(
            self.test['Weather_Condition'].str.contains('Dust', case=False, na=False), 1, 0)
        self.test['Weather_Tornado'] = np.where(
            self.test['Weather_Condition'].str.contains('Tornado', case=False, na=False), 1, 0)

        self.test.drop('Weather_Condition', axis=1, inplace=True)

        label_encoding_features = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway',
                                   'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop',
                                   'Civil_Twilight']
        for feature in label_encoding_features:
            self.test[feature] = LabelEncoder().fit_transform(self.test[feature])

    # def show(self, location):

    #     self.test['Start_Lat'] = float(location[0])
    #     self.test['Start_Lng'] = float(location[1])
    #     self.get_time(location)
    #     self.get_POI(location)
    #     self.get_weather(location)
    #     self.test.drop(
    #         ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station',
    #          'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop', 'Civil_Twilight'], axis=1)
    #     df = self.test
    #     dic = df.to_dict()
    #     for k,v in dic.items():
    #         dic[k] = v[0]
    #     data = {
    #         'Temp':str(dic['Temperature(F)'])+" F",
    #         'Humi':str(dic['Humidity(%)'])+" %",
    #         'Time':str(dic["Year"])+"/"+str(dic["Month"])+"/"+str(dic["Day"])+" "+str(dic["Hour"])+":"+str(dic["Minute"]),
    #         'Visi':str(dic["Visibility(mi)"])+" mi",
    #     }
    #     return data

    def predict(self, location):  # location = [Lat, Lng, CITY]

        # 'Severity', 'Start_Lat', 'Start_lng', 'City',
        self.test['Start_Lat'] = float(location[0])
        self.test['Start_Lng'] = float(location[1])
        self.get_time(location)
        self.get_POI(location)
        self.get_weather(location)
        df = self.test
        dic = df.to_dict()
        self.encoding()
        result = self.model.predict(self.test)
        self.test.drop(
            ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station',
             'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop', 'Civil_Twilight'], axis=1)
        for k,v in dic.items():
            dic[k] = v[0]
        def timeFormat(time):
            if len(str(time))<2: 
                return '0'+str(time)
            else:
                return str(time) 
        data = {
            'Temp':str(dic['Temperature(F)'])+" F",
            'Humi':str(dic['Humidity(%)'])+" %",
            'Time':str(dic["Year"])+"/"+timeFormat(dic["Month"])+"/"+timeFormat(dic["Day"])+" "+timeFormat(dic["Hour"])+":"+timeFormat(dic["Minute"]),
            'Visi':str(dic["Visibility(mi)"])+" mi",
            'Weather': str(dic['Weather_Condition']),
            'Severity': SEVERITY_DICT[result[0]] 
        }
        return data


if __name__ == '__main__':
    obj = Backend()
    data = obj.predict([39.10266, -84.0628])
    print(data)
    # print(obj.predict([39.10266, -84.0628]))

