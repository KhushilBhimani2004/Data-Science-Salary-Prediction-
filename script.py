import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import time
import io
from sklearn.model_selection import train_test_split
#regression models
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge,Lasso,RidgeCV,LassoCV,ElasticNet,ElasticNetCV,LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import neighbors
from sklearn.svm import SVR
from sklearn import metrics

from sklearn.metrics import accuracy_score
import collections
import joblib
import warnings
warnings.filterwarnings("ignore")
# st.table(df)
df1 = 0
gr = GradientBoostingRegressor()
data = pd.read_csv("D:\VIT 5 semester\IP\Mini Project\ds_salaries.csv")
df = data
rad =st.sidebar.radio("Navigation",["Home", "Visualization", "Prediction Models", "Prediction", "About Us"])
if rad == "Home":
    st.markdown("<h2>Dataset :</h2>", True)
    if st.checkbox("Show Dataset: "):
        st.dataframe(data)
        df = pd.DataFrame(data = data)

    st.markdown("Exploratory Data Analysis:")
    if st.checkbox("Show EDA: "):
        st.markdown("<h4>Information about data: </h4>", True)
        buffer = io.StringIO()
        data.info(buf=buffer)
        st.text(buffer.getvalue())
        # st.text(s)
        st.markdown("<h4>Statistics of data: </h4>", True)
        st.write(data.describe(include='all'))
        col1, col2, col3 = st.columns(3) 

        with col1:
            st.write("Data Types of Column Values:",data.dtypes)

        with col2:
            st.write("Null Data:",data.isnull().sum())

        with col3:
            st.write("Unique Values:", data.nunique())

if rad == "Visualization":
    selected_experiences = st.multiselect('Select Experience Levels', df['experience_level'].unique())

    # Filter the DataFrame based on the selected experience levels
    filtered_df = df[df['experience_level'].isin(selected_experiences)]

    # Create a countplot using Plotly Express
    fig = px.bar(filtered_df, x='experience_level', title=f'Experience Level {" ,".join(selected_experiences)}.')
    fig.update_xaxes(title_text='Experience Level')
    fig.update_yaxes(title_text='Count')

    # Display the interactive plot
    st.plotly_chart(fig)


#average salary by work year
    df = pd.DataFrame(data)

# Group the data by work year and calculate the average salary
    df1 = df.groupby("work_year")["salary"].mean().reset_index()

    # Streamlit app
    st.write('Interactive Average Salary by Work Year')

    # Create a bar chart using Plotly Express
    fig = px.bar(df1, x='work_year', y='salary')

    # Remove decimal points from work year values on the x-axis
    fig.update_xaxes(
        title_text='Work Year',
        tickvals=df1['work_year'].astype(int),
        ticktext=df1['work_year'].astype(int)
    )

    fig.update_yaxes(title_text='Average Salary')

    # Display the interactive plot
    st.plotly_chart(fig)



#

    # Plot 2: Salary
    st.subheader('salary')
    fig2 = px.bar(df, y='salary', title='Salary')
    st.plotly_chart(fig2)

    # Plot 3: Salary in USD
    st.subheader('salary_in_usd')
    fig3 = px.bar(df, y='salary_in_usd', title='Salary in USD')
    st.plotly_chart(fig3)

    # Plot 4: Remote Ratio
    st.subheader('remote_ratio')
    fig4 = px.bar(df, y='remote_ratio', title='Remote Ratio')
    st.plotly_chart(fig4)


    #
    
# Create a pie chart using Plotly Express
    # print("Total value counts of the roles:-\n ",df["experience_level"].value_counts())
    roles = ["SE", "MI", "EN", "EX"]
    people = [2516, 805, 320, 114]
    color_scale = ['#ADD8E6', '#87CEEB', '#6495ED', '#4169E1']

    # Streamlit app
    # st.title('Interactive Distribution of Roles')

    # Create a pie chart using Plotly Express
    fig = px.pie(values=people, names=roles, title='Distribution of Roles', labels=roles, color_discrete_sequence=color_scale)

    # Display the interactive plot
    st.plotly_chart(fig)


    #
    # print(df["employment_type"].value_counts())
    types = ["FT", "PT", "CT", "FL"]
    no_people = [3718, 17, 10, 10]

    # Specify a custom color palette for the pie chart
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']

    # Streamlit app
    # st.title('Interactive Number of People by Employment Type')

    # Create a pie chart using Plotly Express
    fig = px.pie(
        values=no_people,
        names=types,
        title='Number of People by Employment Type',
        color_discrete_sequence=colors,
        hole=0.3  # Add a hole in the middle for a donut chart effect
    )

    # Display the interactive plot
    st.plotly_chart(fig)



    # 
    # df["company_size"].value_counts()
    company_numbers = [3153, 454, 148]
    company_size = ["M", "L", "S"]
    explode = [0.1, 0, 0]



    # Create a pie chart using Plotly Express
    fig = px.pie(
        values=company_numbers,
        names=company_size,
        title='Company Size',
        hole=0.3,  # Add a hole in the middle for a donut chart effect
        labels={'names': 'Company Size'},
        height=400  # Set the height of the chart
    )


    # Calculate mean salaries for each job title
    # job_title_salaries = df.groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=False)

    # # Create horizontal bar chart
    # fig, ax = plt.subplots(figsize=(8, 20))
    # ax.barh(job_title_salaries.index, job_title_salaries.values)
    # ax.set_title('Average Salaries by Job Title')
    # ax.set_xlabel('Salary in USD')
    # ax.set_ylabel('Job Title')
    # plt.show()
    # # Display the interactive plot
    # st.plotly_chart(fig)

    #
    job_title_salaries = df.groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=False)

    # Create a horizontal bar chart using Plotly Express
    fig = px.bar(
    job_title_salaries,
    y='salary_in_usd',
    x=job_title_salaries.index,
    orientation='v',  # Horizontal bar chart
    title='Average Salaries by Job Title',
    labels={'salary_in_usd': 'Salary in USD', 'index': 'Job Title'},
    width=800,  # Adjust chart width to accommodate all labels
    height=800  # Adjust chart height to accommodate all labels
    )

    # Display the interactive plot
    st.plotly_chart(fig)




    st.subheader('Salary vs. Work Year by Experience Level')
    catplot1 = px.box(df, x="work_year", y="salary_in_usd", color="experience_level", title="Salary vs. Work Year by Experience Level")
    st.plotly_chart(catplot1)

    # Create an interactive catplot for Salary vs. Work Year by Employment Type
    st.subheader('Salary vs. Work Year by Employment Type')
    catplot2 = px.box(df, x="work_year", y="salary_in_usd", color="employment_type", title="Salary vs. Work Year by Employment Type")
    st.plotly_chart(catplot2)
    
    st.subheader('Salary vs. Company Size')
    catplot2 = px.box(df, x="company_size", y="salary_in_usd", color="company_size", title="Salary vs. Work Year by Employment Type")
    st.plotly_chart(catplot2)

    #
    
# Create a correlation heatmap
    cat_list=[i for i in df.select_dtypes("object")]
    for i in cat_list:
        df[i] = df[i].factorize()[0]

    st.title('Interactive Correlation Heatmap')
    fig = px.imshow(df.corr(), color_continuous_scale='viridis')

    # Customize the figure layout
    fig.update_layout(title="Correlation Heatmap", width=700, height=700)

    # Display the interactive heatmap
    st.plotly_chart(fig)
# X=df.drop(["salary_in_usd"], axis = 1)
# Y=df["salary_in_usd"]
if rad == "Prediction Models":
    X=df.drop(["salary_in_usd"], axis = 1)
    Y=df["salary_in_usd"]
    cat_list=[i for i in df.select_dtypes("object")]
    for i in cat_list:
        df[i] = df[i].factorize()[0]
    X=df.drop(["salary_in_usd"], axis = 1)
    Y=df["salary_in_usd"]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    st.markdown(f"X Train: {X_train.shape},  X Test: {X_test.shape}, Y Train {Y_train.shape} ,Y Test{Y_test.shape}")
    st.markdown("<h3>Decision Tree Regressor Model", True)
    dt=DecisionTreeRegressor()
    dt.fit(X_train,Y_train)
    DecisionTreeRegressor()
    y_predict = dt.predict(X_test)
    st.write("Train Model Score: ",dt.score(X_train,Y_train))
    st.write("Test Model Score: ",dt.score(X_test,Y_test))

    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    st.write("R2 Score: ",r2_score(Y_test, y_predict)*100)
    st.write("Mean Squared Error: ",mean_squared_error(Y_test, y_predict))
    st.write("Mean Absolute Error: ",mean_absolute_error(Y_test, y_predict))

    knn=KNeighborsRegressor().fit(X_train,Y_train)
    ada=AdaBoostRegressor().fit(X_train,Y_train)
    svm=SVR().fit(X_train,Y_train)
    ridge=Ridge().fit(X_train,Y_train)
    lasso=Lasso().fit(X_train,Y_train)
    rf=RandomForestRegressor().fit(X_train,Y_train)
    gbm=GradientBoostingRegressor().fit(X_train,Y_train)

    models=[ridge,lasso,knn,ada,svm,rf,gbm]

    def ML(Y,models):
        y_pred=models.predict(X_test)
        mse=mean_squared_error(Y_test,y_pred)
        rmse=np.sqrt(mean_squared_error(Y_test,y_pred))
        r2=r2_score(Y_test,y_pred)*100
        
        return mse,rmse,r2

    for i in models:
        st.write("\n",i,"\n\nDifferent models success rate :",ML("salary_in_usd",i))

if rad=="Prediction":
    

    # st.write(df['salary_currency'].unique())
    data = pd.read_csv("D:\VIT 5 semester\IP\Mini Project\ds_salaries.csv")
    cat_list=[i for i in df.select_dtypes("object")]
    for i in cat_list:
        df[i] = df[i].factorize()[0]
    columns_to_drop = ['salary_in_usd', 'salary']
    X=df.drop(columns_to_drop, axis=1)
    # X=df.drop([""])

    # st.dataframe(X)
    Y=df["salary_in_usd"]
    dt=DecisionTreeRegressor()
    dt.fit(X,Y)
    # st.title("Data Science Salary Predictor:")

    work_year = st.selectbox("Choose work year: ", ['2022','2021','2022','2023'])
    exp_level = st.radio(
    "What\'s Experience Level:",
    ('SE','MI', 'EN', 'EX'))

    if exp_level == 'SE':
        elvl = 0
    elif exp_level == 'ME':
        elvl = 1
    elif exp_level == 'EN':
        elvl = 2
    else:
        elvl = 3
        
    emp_type = st.radio(
    "What\'s Employment Type:",
    ('FT','CT', 'FL', 'PT'))

    if emp_type == 'FT':
        etype = 0
    elif emp_type == 'CT':
        etype = 1
    elif emp_type == 'FL':
        etype = 2
    else:
        etype = 3
    

    
    job_titles = [
        "Principal Data Scientist",
        "ML Engineer",
        "Data Scientist",
        "Applied Scientist",
        "Data Analyst",
        "Data Modeler",
        "Research Engineer",
        "Analytics Engineer",
        "Business Intelligence Engineer",
        "Machine Learning Engineer",
        "Data Strategist",
        "Data Engineer",
        "Computer Vision Engineer",
        "Data Quality Analyst",
        "Compliance Data Analyst",
        "Data Architect",
        "Applied Machine Learning Engineer",
        "AI Developer",
        "Research Scientist",
        "Data Analytics Manager",
        "Business Data Analyst",
        "Applied Data Scientist",
        "Staff Data Analyst",
        "ETL Engineer",
        "Data DevOps Engineer",
        "Head of Data",
        "Data Science Manager",
        "Data Manager",
        "Machine Learning Researcher",
        "Big Data Engineer",
        "Data Specialist",
        "Lead Data Analyst",
        "BI Data Engineer",
        "Director of Data Science",
        "Machine Learning Scientist",
        "MLOps Engineer",
        "AI Scientist",
        "Autonomous Vehicle Technician",
        "Applied Machine Learning Scientist",
        "Lead Data Scientist",
        "Cloud Database Engineer",
        "Financial Data Analyst",
        "Data Infrastructure Engineer",
        "Software Data Engineer",
        "AI Programmer",
        "Data Operations Engineer",
        "BI Developer",
        "Data Science Lead",
        "Deep Learning Researcher",
        "BI Analyst",
        "Data Science Consultant",
        "Data Analytics Specialist",
        "Machine Learning Infrastructure Engineer",
        "BI Data Analyst",
        "Head of Data Science",
        "Insight Analyst",
        "Deep Learning Engineer",
        "Machine Learning Software Engineer",
        "Big Data Architect",
        "Product Data Analyst",
        "Computer Vision Software Engineer",
        "Azure Data Engineer",
        "Marketing Data Engineer",
        "Data Analytics Lead",
        "Data Lead",
        "Data Science Engineer",
        "Machine Learning Research Engineer",
        "NLP Engineer",
        "Manager Data Management",
        "Machine Learning Developer",
        "3D Computer Vision Researcher",
        "Principal Machine Learning Engineer",
        "Data Analytics Engineer",
        "Data Analytics Consultant",
        "Data Management Specialist",
        "Data Science Tech Lead",
        "Data Scientist Lead",
        "Cloud Data Engineer",
        "Data Operations Analyst",
        "Marketing Data Analyst",
        "Power BI Developer",
        "Product Data Scientist",
        "Principal Data Architect",
        "Machine Learning Manager",
        "Lead Machine Learning Engineer",
        "ETL Developer",
        "Cloud Data Architect",
        "Lead Data Engineer",
        "Head of Machine Learning",
        "Principal Data Analyst",
        "Principal Data Engineer",
        "Staff Data Scientist",
        "Finance Data Analyst"

    ]
    job_title_to_number = {}

# Assign unique numbers to job titles
    for i, title in enumerate(job_titles, start=0):
        job_title_to_number[title] = i

    # Allow the user to select a job title
    selected_title = st.selectbox("Select a Job Title:", job_titles)

    # Find the unique number for the selected title
    unique_number = job_title_to_number.get(selected_title, None)

    if unique_number is not None:
        st.write(f"Selected Job Title: {selected_title}")
        # st.write(f"Unique Number: {unique_number}")
    else:
        st.write("Job title not found in the list.")


    # 
    salary_currency_values = [
    "EUR", "USD", "INR", "HKD", "CHF", "GBP", "AUD", "SGD", "CAD", "ILS", "BRL", "THB", "PLN", "HUF", "CZK", "DKK",
    "JPY", "MXN", "TRY", "CLP", # Add more currency values
    ]

    # Create a dictionary to map salary currency values to unique numbers
    currency_to_number = {value: i for i, value in enumerate(salary_currency_values)}

    # st.title('Salary Currency Numbering Tool')

    # User input for salary currency value using a select box
    user_input = st.selectbox('Select a salary currency value:', salary_currency_values)

    # Function to assign a unique number to the currency value
    def assign_currency_number(currency_value):
        if currency_value in currency_to_number:
            return currency_to_number[currency_value]
        else:
            return "Currency value not found"

    # if st.button('Assign Number'):
    currency_number = assign_currency_number(user_input)
    # st.write(f'Salary Currency Value: {user_input}, Unique Number: {currency_number}')

    # Display the unique salary currency values and their corresponding numbers
    # st.subheader('Unique Salary Currency Values and Numbers:')
    # 

    
    employee_residence_values = [
    "ES", "US", "CA", "DE", "GB", "NG", "IN", "HK", "PT", "NL", "CH", "CF", "FR", "AU", "FI", "UA", "IE", "IL", "GH", "AT", # Add more residence values
    ]

    # Create a dictionary to map employee residence values to unique numbers
    residence_to_number = {value: i for i, value in enumerate(employee_residence_values)}

    # st.title('Employee Residence Numbering Tool')

    # User input for employee residence value
    # user_input = st.text_input('Enter an employee residence value:')
    user_input = st.selectbox('Select an employee residence value:', employee_residence_values)

    # Function to assign a unique number to the residence value
    def assign_residence_number(residence_value):
        if residence_value in residence_to_number:
            return residence_to_number[residence_value]
        else:
            return "Residence value not found"

    # if st.button('Assign Number'):
    residence_number = assign_residence_number(user_input)
    # st.write(f'Employee Residence Value: {user_input}, Unique Number: {residence_number}')

    # Display the unique employee residence values and their corresponding numbers
    # st.subheader('Unique Employee Residence Values and Numbers:')
    # st.write(pd.DataFrame(list(residence_to_number.items()), columns=['Employee Residence', 'Unique Number']))

    rem_ratio = st.radio(
    "Enter Remote Ratio:",
    ('Remotely','On-Site', 'Hybrid'))

    if rem_ratio == 'Remotely':
        rr = 100
    elif rem_ratio == 'On-Site':
        rr = 0
    else:
        rr = 50

    company_location_values = [
    "ES", "US", "CA", "DE", "GB", "NG", "IN", "HK", "NL", "CH", "CF", "FR", "FI", "UA", "IE", "IL", "GH", "CO", "SG", "AU", "SE", "SI", "MX", "BR", "PT", "RU", "TH", "HR", "VN", "EE", "AM", "BA", "KE", "GR", "MK", "LV", "RO", "PK", "IT", "MA", "PL", "AL", "AR", "LT", "AS", "CR", "IR", "BS", "HU", "AT", "SK", "CZ", "TR", "PR", "DK", "BO", "PH", "BE", "ID", "EG", "AE", "LU", "MY", "HN", "JP", "DZ", "IQ", "CN", "NZ", "CL", "MD", "MT", # Add more company location values
    ]

    # Create a dictionary to map company location values to unique numbers
    location_to_number = {value: i for i, value in enumerate(company_location_values)}

    # st.title('Company Location Numbering Tool')

    # User input for company location value using a select box
    user_input = st.selectbox('Select a company location value:', company_location_values)

    # Function to assign a unique number to the location value
    def assign_location_number(location_value):
        if location_value in location_to_number:
            return location_to_number[location_value]
        else:
            return "Location value not found"

    # if st.button('Assign Number'):
    location_number = assign_location_number(user_input)
    # st.write(f'Company Location Value: {user_input}, Unique Number: {location_number}')

    # Display the unique company location values and their corresponding numbers
    # st.subheader('Unique Company Location Values and Numbers:')

    co_size = st.radio(
    "What\'s Comapny Size:",
    ('Small','Large', 'Medium'))

    if co_size == 'Large':
        c_size = 0
    elif co_size == 'Medium':
        c_size = 2
    else:
        c_size = 1

    if st.button('PREDICT'):
        joblib.dump(dt,'model_train') #training model using joblib 
        model = joblib.load('model_train')
        result  = model.predict([[work_year, elvl, etype, unique_number, currency_number, residence_number,rr, location_number, c_size]])
        st.success(f'Insurance Cost: $ {result[0]}')
        # st.write('Insurance Cost: $', result[0])
    else:
        st.error('Some Error Occurred', icon="☠️")

if rad == "About Us":

    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.02)
        progress.progress(i+1)    

    st.snow()
    st.title("About Us")

    st.markdown(
            """
            <style>
            .container {
                display: flex;
                justify-content: space-around;
                align-items: center;
                margin-bottom: 50px;
            }
            
            .person {
                text-align: center;
                padding: 20px;
            }
            
            .person h3 {
                margin-bottom: 10px;
                margin-top: -10px;
            }
            
            .person p {
                font-size: 18px;
                line-height: 1.5;
            }
            </style>
            """,True)


    st.markdown(
            """
            <div class="container">
                <div class="person">
                    <h3>Khushil Bhimani</h3>
                        <h5>B.E in Computer Engineering<br>
                        College Name: Vidyalankar Institute of Technology</h5>
                    <strong>Contact Information:</strong>
                    <ul>
                        <li>Email: khushilbhimani2@gmail.com.com</li>
                        <li>Phone No.:9324130035</li>
                        </ul>
                        </div>
            """,True)

    st.markdown(
            """
            <div class="container">
                <div class="person">
                    <h3>Chinmay Mhatre</h3>
                    <h5>B.E in Computer Engineering<br>
                        College Name: Vidyalankar Institute of Technology</h5>
                    <strong>Contact Information:</strong>
                    <ul>
                        <li>Email: chinmaymhatre@gmail.com</li>
                        <li>Phone No.:9324339904</li>
                    </ul>
                </div>
            """,True)

    st.markdown(
            """
            <div class="container">
                <div class="person">
                    <h3>Atharva Ingole</h3>
                    <h5>B.E in Computer Engineering<br>
                        College Name: Vidyalankar Institute of Technology</h5>
                    <strong>Contact Information:</strong>
                    <ul>
                        <li>Email: atharvaingole@gmail.com</li>
                        <li>Phone No.:8080718581</li>
                    </ul>
                </div>
            """,True)



