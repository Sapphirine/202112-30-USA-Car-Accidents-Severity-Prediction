# USA Car Accidents Severity Prediction 

The goal of our project is to built an application to visualize analyzed data of USA car accidents and predict the severity level of one car accident according to features like time, road, weather, etc.

### Group Members: 
Zifan Chen (zc2628)
Meiyou Liu (ml4687)
Yuxing Wang (yw3739)

# Project Overview

### System Architecture
The whole system was built via Django framework. Our team used JavaScript, CSS (templet from html5up.net), jQuery, and HTML5 to construct our front-end which is responsible for visualizing the analyzed data, interacting with users, and sending users’ prediction requests to the back-end via ajax GET request. After getting the json data from the back-end, the front end will use these data to form a new HTML table to replace old HTML table content so that users get their required information without refreshing the website page.

For the back-end, our team decode the url and get longitude and latitude from it, then used API requests to gather time, road, and weather features and feed them to the prediction model. The prediction result and some key features will be sent to front-end in the form of json. 

### Dataset
The dataset we used in our project is contributed by a Lyft scientist, Sobhan Moosavi. It is a countrywide traffic accident dataset, which covers 49 states of the United States, includes more than 3 million accident records and 46 features.

### Data Preprocessing
We visualize the data according to different features. Then we dropped 27 features because of their low correlation with severity in correlation matrix. 

### Model Training And Choosing
We used the left features to train Logistic regression, SVM, Decision Tree, Gradient Boost Tree, Random Forest, Multi-layer Perceptron Classification, and compare their quality according to their accuracy, precision, recall, etc. Finally, we chose Decision Tree as the optimal model we will use in our system. We used 75% of the data as training set and the left 25% as testing set. The accuracy of Decision Tree model is 87%, which is highest among all the model we trained.

# Setting Up And Use Our Application

1. Download the whole final_project directory
2. Start the system server
    1. Install Django (https://docs.djangoproject.com/en/4.0/topics/install/#installing-official-release)
    2. Open this directory and open a cmd terminal there.
    3. Try to deploy the server by using the command below:
        ```
        python manage.py runserver 
        ```
        If you get any 
        `ImportError: No module named <modual name>`
        please use the following command:
        ```
        pip install <modual name>
        ```
        Packages we used and you may need to install are listed below:
        ```
        overpy datetime time tzwhere pytz pandas requests json geopy numpy sklearn joblib
        ```
        untill you see something like:
        ```
        Watching for file changes with StatReloader
        Performing system checks...
        
        System check identified no issues (0 silenced).
        December 22, 2021 - 21:25:33
        Django version 3.2.8, using settings 'final.settings'
        Starting development server at http://<IP Address>:<port>/
        Quit the server with CTRL-BREAK.
        ```
    4. Use our application via browser
        Please use <IP Address> and <port> you get from above section and enter the following URL in the browser URL bar to use our application:
        ```
        http://<IP Address>:<port>/homepage/
        ```
        1. Abstruct of our project at the top of the website. A navigation bar is on the left side, where you could click and be navigated to the place you want to go.
        
        2. For the data visualization section, we analyzed the data and visualize them in the form of histograms and pie charts according to different features. There are charts for Location Feature, Road Feature, Weather Feature, and Time Feature.
        
        3. For the prediction section. Users could type in the longitude and latitude of the place they want to predict ``(please only use [lon, lat] inside USA, otherwise you will get an UnknownTimeZoneError alert)``. After clicking the submit button and waiting for few seconds, a table of key features' information and prediction result will be updated and present.


# Repository Content
```
│   README.md
│
├───data_and_model
│       data_analysis.ipynb
│       data_preprocess_and_training.ipynb
│       preprocessed_data.csv
│       US_Accidents_Dec20_updated.csv
│
└───final_project
    │   db.sqlite3
    │   manage.py
    │
    ├───final
    │   │   asgi.py
    │   │   dt.pkl
    │   │   predict.py
    │   │   settings.py
    │   │   urls.py
    │   │   view.py
    │   │   wsgi.py
    │   │   __init__.py
    │   │
    │   └───__pycache__
    │           predict.cpython-36.pyc
    │           settings.cpython-36.pyc
    │           urls.cpython-36.pyc
    │           view.cpython-36.pyc
    │           wsgi.cpython-36.pyc
    │           __init__.cpython-36.pyc
    │
    ├───static
    │   ├───css
    │   │   │   font-awesome.min.css
    │   │   │   main.css
    │   │   │
    │   │   └───images
    │   │       │   banner.jpg
    │   │       │   overlay.png
    │   │       │
    │   │       └───ie
    │   │               grad0-15.svg
    │   │
    │   ├───fonts
    │   │       fontawesome-webfont.eot
    │   │       fontawesome-webfont.svg
    │   │       fontawesome-webfont.ttf
    │   │       fontawesome-webfont.woff
    │   │       fontawesome-webfont.woff2
    │   │       FontAwesome.otf
    │   │
    │   ├───images
    │   │       .DS_Store
    │   │       avatar.jpg
    │   │       banner.jpg
    │   │       L1.png
    │   │       L2.png
    │   │       pic02.jpg
    │   │       pic03.jpg
    │   │       pic04.jpg
    │   │       pic05.jpg
    │   │       pic06.jpg
    │   │       pic07.jpg
    │   │       pic08.jpg
    │   │       R1.png
    │   │       T1.png
    │   │       T2.png
    │   │       T3.png
    │   │       T4.png
    │   │       W1.png
    │   │
    │   ├───js
    │   │   │   jquery.min.js
    │   │   │   jquery.scrolly.min.js
    │   │   │   jquery.scrollzer.min.js
    │   │   │   main.js
    │   │   │   skel.min.js
    │   │   │   util.js
    │   │   │
    │   │   └───ie
    │   │           backgroundsize.min.htc
    │   │           html5shiv.js
    │   │           PIE.htc
    │   │           respond.min.js
    │   │
    │   ├───model
    │   └───sass
    │       │   ie8.scss
    │       │   ie9.scss
    │       │   main.scss
    │       │
    │       └───libs
    │               _functions.scss
    │               _mixins.scss
    │               _skel.scss
    │               _vars.scss
    │
    └───templates
            index.html
```

### data_and_model
This directory contains 
```
    US_Accidents_Dec20_updated.csv      -> original dataset
    preprocessed_data.csv               -> preprocessed dataset
    data_analysis.ipynb                 -> jupyter notebook that analyzes data and generates diagrams
    data_preprocess_and_training.ipynb  -> jupyter notebook that generate different models and store the best one
```
### final_project
This directory is the whole Django project.
Key files：
```
    final_project/final/dt.pkl          -> decision tree model file
    final_project/final/predict.py      -> py file that unpacks model file, requests API information, predicts the severity level with [lon, lat], and return key API features and prediction result
    final_project/final/urls.py         -> py file to process url request
    final_project/final/view.py         -> py file to accept url request and return content to that request
    final_project/static/images         -> directory that store pictures our website needs
    final_project/templates/index.html  -> html file that builds the front-end of our project
```

