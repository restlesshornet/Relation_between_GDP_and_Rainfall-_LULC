import streamlit as st
import streamlit.components.v1 as components 
import numpy as np 
import streamlit_option_menu as option_menu
from PIL import Image, ImageDraw
import io
import tempfile
import os
import shutil
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
        
def main():
    with open("styles.css", "r") as source_style:
        st.markdown(f"<style>{source_style.read()}</style>", 
             unsafe_allow_html = True)
        
    st.title("IISF Space India Hackathon'2023")
    Header = st.container()
    
    ##MainMenu
    
    with st.sidebar:
        selected = option_menu.option_menu(
            "Main Menu",
            options=[
                "Project Information",
                "Model Information",           
                "Data Visualisation",
                "Our Team"
            ],
        )
    
    st.sidebar.markdown('---')
        
    ##HOME page 
    
    if selected == "Project Information":
        st.image("logo.png")
        st.subheader("Team Name : Odyssey")
        st.subheader("Group : 2 - Topic : 2")
        st.subheader("Problem Statement")
        problem_statement = """
        <div style="text-align: justify;">
        The project involves unveiling hidden parameters through diverse Machine Learning Approaches to establish a robust relationship
        between <b>Land Use and Land Cover (LULC)</b> and <b>Gross Domestic Product (GDP)</b>. Integration of atmospheric and geophysical data is
        essential to obtain comprehensive correlations across all datasets. The developed models should output spatial-temporal hotspots
        and trends concurrently from multiple datasets, with a focus on anomaly validation and further in-depth analysis if needed.
        </div>
        """
        st.markdown(problem_statement, unsafe_allow_html=True)
        st.subheader("Our Solution")
        Project_goal = """
        <div style="text-align: justify;">
        Our project begins with data collection from Bhuvan's LULC dataset, IMD's rainfall dataset, and RBI's GDP data. The process
involves converting raster data to CSV through vectorization, allowing isolation of each LULC feature. <b>QGIS and Rasterio</b> Library are
then employed to gain deeper insights into Geotiff and explore associated statistics.
Data anomalies are addressed through data visualization using <b>Matplotlib</b>, <b>Plotly</b>, and <b>Seaborn</b>. Further analysis is conducted using
authoritative government documents for research and statistical validation.
For model training, we leverage <b>TabNet, Linear Regression, Multi-Layer Perceptron, LGBM Regression, and XGB Regression</b> to
unveil patterns within GDP, LULC, and atmospheric data. These models aid in identifying hotspots, thereby revealing robust
relationships between the variables.
        </div>
        """
        st.markdown(Project_goal, unsafe_allow_html=True)

        st.subheader("Detailed Proposal & Solution Approach")
        st.image("IISF WorkFlow.png")
        Solution = """
        <div style="text-align: justify;">
        1. <b>Data Collection</b> : Utilize Bhuvan's LULC dataset, IMD's rainfall 
            dataset, and RBI's GDP data for comprehensive coverage.
        </div>
        """
        Soultionb = """
        <div style="text-align: justify;">
2. <b>Data Preprocessing</b> : For the Preprocessing of the data we plan to 
follow through the following steps : \n
- Convert raster data to CSV through vectorization, ensuring 
isolation of individual LULC features. \n
- Employ QGIS and Rasterio Library to extract additional 
information from the Geotiff files and explore statistical Insights
    </div>
        
        """
        Soultionc = """
        <div style="text-align: justify;">\n
3. <b>Data Visualization for Anomalies</b>: Use Matplotlib, Plotly, and 
Seaborn for data visualization to identify and address anomalies.\n
4. <b>Further Analysis with Authorized Data Sources</b>: Refer to 
government documents for additional research and statistical 
validation to enhance the reliability of the findings.\n
5. <b>Model Training</b>: Utilize a diverse set of models for training, including 
TabNet, Linear Regression, Multi-Layer Perceptron, LGBM 
Regression, and XGB Regression.\n
6. <b>Unveiling Data Patterns:</b>Apply the trained models to unveil 
patterns within GDP, LULC, and atmospheric data.\n
7. <b>Identification of Hotspots:</b> The models contribute to identifying 
spatio
-temporal hotspots, indicating robust relationships between 
different variables.\n
8. <b>Results Visualization:</b> Create visual representations of the 
results using appropriate tools to ensure clarity and interpretability.\n
9. <b>Anomaly Validation and Further Analysis:</b> If anomalies are 
detected, validate and investigate the reasons behind them. 
Conduct further analysis if required for a more in
-depth
understanding.\n
10. <b>Iterative Optimization:</b> Fine
-tune models and analysis based 
on feedback, ensuring continuous improvement and optimization.\n
11. <b>Documentation and Reporting:</b> Document the entire process, 
methodologies, and findings. Prepare comprehensive reports for 
transparent communication of results.
</div>

        """
        

        st.markdown(Solution, unsafe_allow_html=True)
        st.markdown(Soultionb, unsafe_allow_html=True)
        st.image("LULC.png")
        st.markdown(Soultionc, unsafe_allow_html=True)

        st.subheader("Tools & Devices used on Development")
        devices = """
        <div style="text-align: justify;">\n
            • QGIS.\n
            • Rasterio & Similar Libraries.\n
            • Neural networks (CNNs) for Relationship Analysis.\n
            • TensorFlow, PyTorch for implementing Dynamic ML Algo etc.\n
            • GitHub for version control and documentation.\n
        </div>
        """
        st.markdown(devices, unsafe_allow_html=True)
        
        st.subheader("Technologies Involved/used")
        tech = """
        <div style="text-align: justify;">\n
            • TIF files & Spatial Analysis.\n
            • Image Processing Libraries.\n
            • Data Analysis & validation.\n
            • Deep Learning Framework.\n
            • Software for Tempo-Spatial Area.\n
            • Collaboration platforms.\n
        </div>
        """
        st.markdown(tech, unsafe_allow_html=True)

        st.subheader("References")
        references = """
        1. https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
        2. https://rbi.org.in/Scripts/PublicationsView.aspx?id=22091
        3. https://bhoonidhi.nrsc.gov.in/bhoonidhi/home.html
        4. https://pytorch.org/
        """

        st.markdown(references, unsafe_allow_html=True)
    elif selected == "Model Information": 
        st.image("logo.png")
        st.subheader("Model Information")
        Intro = """
        <div style="text-align: justify;">\n
While trying to unveil the hidden parameters and establishing a robust relationship between the Land Use and Land Cover ( LULC) and the Gross Domestic Product (GDP), we have explored a variety of Machine Learning & Deep Neural Networks in order to understand 
the comprehensive correlations across the atmospheric and the geophysical data.
These models are :\n
1. TabNet 
2. LightGBM 
3. XGboost Regressor 
4. SVR (Support Vector Regressor) 
5. MLP ( Multi Layer perceptron ) 

</div>
        """
        st.markdown(Intro, unsafe_allow_html=True)

        st.subheader("1. TabNet")
        TabNet = """
        <div style="text-align: justify;">
        TabNet or Tabular Nueral Network is a unique Deep Nueral Network that is designed to address the tabular data. It employees a tree based learning approach for it's training process, TabNet Can perform representation learning on the unlabeled data. Morever,
        it uses schotastic based gradient descent method to perform smoothly even when the dataset is large. TabNet follows the attention based mechanism to selectively emphasize relevant features during the decision-making process and thus allowing  the model to 
        handle complex relationships within the dataset, making it well-suited for our task of correlating LULC and GDP.\n
        <b>Architecture</b> : The TabNet Architecture Basically consists of a multi-steps in a sequential manner passing the input from one step to another where there are different ways to decide the number of steps depending upon the capacity. Each step consists of the following steps:
        Where in the initial step, the complete dataset is input into the model without any feature engineering. It is then passed through a batch normalization layer, and after that, it is then passed in a feature transformer which basically consists of n number of different
        GLU Blocks where each GLU Block consists of a Fully connected Layer, Batch Normalization layer and a Gated Linear Unit.
        </div>
        """
        st.markdown(TabNet, unsafe_allow_html=True)
        st.image('feattransformer-660x217.jpeg')
        Results = """
        <div style="text-align: justify;">
        According to the observations while training the model we observed a cv-score of 0.63 , which is a decent score keeping in mind that we had a limitation with the dataset, however this could also indicate towards the overfitting of the data. Also a point to note that when we fit the model while excluding the rainfall data which has little to no change in the value.
        </div>
        """
        st.markdown(Results, unsafe_allow_html=True)

        st.subheader("2. LightGBM")
        LightGBM = """
<div style="text-align: justify;">
        LightGBM or Light Gradient Boosting Machine, is a gradient boosting framework that uses the tree-based algorithms, specifically the decision trees which is known for handling large datasets efficiently and train models faster that the traditional gradient boosting framework.\n
It uses two novel techniques:\n

- Gradient-based One Side Sampling(GOSS) 
- Exclusive Feature Bundling (EFB).\n
These techniques fulfill the limitations of the histogram-based algorithm that is primarily used in all GBDT (Gradient Boosting Decision Tree) frameworks.\n
<b>Architecture :</b>  

LightGBM splits the tree leaf-wise as opposed to other boosting algorithms that grow tree level-wise. It chooses the leaf with the maximum delta loss to grow. It builds the tree by expanding the leaves in a way that minimizes the loss the most. Essentially, it's like choosing the best path to explore in the tree.

This leaf-wise strategy can be advantageous because it often results in lower loss compared to the traditional level-wise method. However, it's essential to note that this approach might make the model more complex, potentially causing it to fit the training data too closely. This could be a concern, especially when dealing with smaller datasets, as the model might end up capturing noise in the data, leading to overfitting.
        </div>
"""
        st.markdown(LightGBM, unsafe_allow_html=True)
        st.image("Leaf-Wise-Tree-Growth.png")
        Results = """
        <div style="text-align: justify;">
        According to the observations while training the model we observed a cv-score of 0.354 , which is a low score as compared to what we recieved while using the TabNet Model where it also got a score of 0.269 when we fitted the data in the model by excluding the rainfall part.
        </div>
        """
        st.markdown(Results, unsafe_allow_html=True)

        st.subheader("3. XGBoost Regressor")
        XGB = """
<div style = "text-align: justify;">
XGboost is an ensemble learning method which that combines the predictions of multiple weak models to produce a stronger prediction. One of the key features of XGBoost is its efficient handling of missing values, which allows it to handle real-world data with missing values without requiring significant pre-processing. Additionally, XGBoost has built-in support for parallel processing, making it possible to train models on large datasets in a reasonable amount of time.\n
**Architecture** : In this algorithm, decision trees are created in sequential form. Weights play an important role in XGBoost. Weights are assigned to all the independent variables which are then fed into the decision tree which predicts results. The weight of variables predicted wrong by the tree is increased and these variables are then fed to the second decision tree. These individual classifiers/predictors then ensemble to give a strong and more precise model. It can work on regression, classification, ranking, and user-defined prediction problems.
</div>
"""
        st.markdown(XGB, unsafe_allow_html=True)
        st.image('bagging-sample.png')
        Results = """
                <div style="text-align: justify;">
                According to the observations while training the model we observed a cv-score of 0.45 , which is a low score as compared to what we recieved while using the TabNet Model but a better performance than the LightGBM model where it also got a score of 0.46 when we fitted the data in the model by excluding the rainfall part which shows a slightly better performance.
                </div>
                """
        st.markdown(Results, unsafe_allow_html=True)

        st.subheader("4. SVR (Support Vector Regressor) ")
        SVR = """
<div style = "text-align: justify;">
Support vector regression (SVR) is a type of support vector machine (SVM) that is used for regression tasks. It tries to find a function that best predicts the continuous output value for a given input value.
SVR can use both linear and non-linear kernels. \n
**Architecture** : The architecture of SVR involves creating a hyperplane that represents the best fit to the training data, with the goal of minimizing the error between the predicted and actual values, where it uses a kernel trick to transform the input features into a higher-dimensional space. This transformation enables SVR to capture non-linear relationships among variables. Common kernel functions include radial basis function (RBF), polynomial, and sigmoid.\n
</div>
"""
        st.markdown(SVR, unsafe_allow_html=True)     
        st.image('52329236714_072b75cf80_b.jpg')
        Results = """
                <div style="text-align: justify;">
                According to the observations while training the model we observed a cv-score of 0.22 , which is a very low score as compared to what we recieved while  using the other Model and a further low value of 0.21 when tested without the rainfall data .
                </div>
                """
        st.markdown(Results, unsafe_allow_html=True)

        st.subheader("4. MLP (Multi Layer Perceptron) ")
        SVR = """
<div style = "text-align: justify;">
Multi-layer perception is also known as MLP. It is fully connected dense layers, which transform any input dimension to the desired dimension. A multi-layer perception is a neural network that has multiple layers. To create a neural network we combine neurons together so that the outputs of some neurons are inputs of other neurons. \n
**Architecture** : A multi-layer perceptron has one input layer and for each input, there is one neuron(or node), it has one output layer with a single node for each output and it can have any number of hidden layers and each hidden layer can have any number of nodes. Every node in the multi-layer perception uses a sigmoid activation function. The sigmoid activation function takes real values as input and converts them to numbers between 0 and 1 using the sigmoid formula.\n
</div>
"""
        st.markdown(SVR, unsafe_allow_html=True)     
        st.image('WhatsApp Image 2024-01-18 at 21.22.51_b6a85d0e.jpg')
        Results = """
                <div style="text-align: justify;">
                According to the observations while training the model we observed a cv-score of 0.99 , which is a very high score as compared to what we recieved while  using the other Models but the catch is the following model has a high tendancy to be overfitted due to less data points in the dataset and we can find that there is a similar value when tested without the rainfall data .
                </div>
                """
        st.markdown(Results, unsafe_allow_html=True)
                       
    elif selected == "Our Team":
        st.image("logo.png")
        st.subheader("Our Team")
        st.text("""1.  Aniruddha Kumar\n2.  Akshit Srivastava \n3.  Nikhil Mathur \n4.  Nyasha Rai. \n""")

    elif selected == "Data Visualisation":
        selected_section = st.sidebar.selectbox("Select Section", ["Correlation Matrix", "Crop Distribution","Water & Plantation", "WasteLand & Forest","Grasslands Over Years", "crop seasons", "Built-up"])
        st.subheader('Data Visualisation using MatplotLib & Seaborn')
        df = pd.read_csv("Final_Dataset.csv")
        if selected_section == "Correlation Matrix":
            st.subheader("Correlation matrix")
            fig1, ax1 = plt.subplots(figsize=(8, 8))
            sns.heatmap(df.corr(), fmt='.1f', annot=True, cmap=sns.cubehelix_palette(as_cmap=True), ax=ax1)
            st.pyplot(fig1)

        
        elif selected_section == "Crop Distribution":
            
            st.subheader("Year Wise Crop Distribution")
            feature_names = ['Rabi Crop', 'Kharif Crop', 'Zaid Crop', 'Double/Triple Crop']

            for index, row in df.iterrows():
                data = row[feature_names]
                year = int(row['Time'])
                
                # Create a new figure for each iteration
                fig2, ax2 = plt.subplots(figsize=(8, 8))
                ax2.pie(data, labels=feature_names, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
                ax2.set_title(f'{year}')
                
                # Display the Matplotlib figure using st.pyplot()
                st.pyplot(fig2)
            conclusion = """
            <div style="text-align: justify;">
            Uttar Pradesh is an agrarian economy where 47% of the population is directly dependent on agriculture for their livelihood [Research Paper](https://link.springer.com/chapter/10.1007/978-981-15-9335-2_7)

    Sugarcane Production (Kharif Crop) - 44%
    Rice Production (Kharif Crop) - 12%
    Wheat Production (Rabi Crop) - 28%

    <h3>CONCLUSION</h3>
    
    - Kharif Crop  is showing moderate correlation with the GDP data even though the Kharif Crops - of this season are rice, cotton, jute, sugarcane, arhar, bajra, groundnut, maize, etc.

    - Zaid Crop is showing Very High Negative correlation with GDP data which we also see as the total percentage of Area used to grow Zaid Crops are very less, means that the production of Zaid Crop is also quite low. Hence, we can validate our Conclusion.

    - Rabi Crop has weak positive Correaltion, though it can be towards positive as Wheat is very largly produced in UP So, with more data, we will get strong correlation 

    - Double/Triple Crop has Positive Weak Correaltion or We could say very close to zero. Reason Being - People got to know about the Doublw/Triple Crop Style and were experimenting about that in those year, but it was a failure, <b>As Mentioned in the Research paper</b>, and thus land Use were there, But No contribution to GDP was found.

    <h3>OBSERVATION</h3> 

    - Intially the Kharif Crop was Showing negative correlation with the data, But as the data increased the correlation of the Kharif also increased.
    - Similar, the same conclusion is drawn for the Rabi and Zaid Crops
    - Rabi Crop shows us the trend of taking more than 50% area after 2015, thus creating a stronger correlation could be obtained.


    </div>
            """
            st.markdown(conclusion, unsafe_allow_html=True)  

        elif selected_section == "Water & Plantation":
             
             fig, ax = plt.subplots(figsize=(10, 5))
             df.plot(x='Time', y=['Waterbodies max', 'Waterbodies min', 'Plantation', 'Rainfall'], ax=ax)

            # Set the x-axis label
             plt.xlabel('Year')

            # Set the y-axis label
             plt.ylabel('Data Points')

            # Set the title of the plot
             plt.title('Water Bodies Min & Max with Plantation')

            # Display the plot
             st.pyplot(fig)

             con = """
<div style="text-align: justify;">
<h3>CONCLUSION</h3>
 
- Over the period of time, from a decade, if we observe, then we can say that, use of Tubewell and Wells has been increased comapred to Canal

- Hence, the Overall Decrement of WaterBodies Min can be verified from this Statement.

- And WaterBodies Min is decreasing that's Why it shows negative Correlation with GDP.

- Whereas Increament in  WaterBodies Max is Quite Temprary that's why we have weak Positive Correlation 

- Another Prespective, we can take is that, As More Rainfall Will occur, More Water in Rivers and Lakes. Hence, Large WaterBordies Max Value.



<h4>Plantation</h4> 

- Plantation is highest and positively Correlated with all of these parameters
- More Rainfall more Plantation, Higher Land Use, Good Health of plantation. Thus, moderate to high Positive Correlation with all the Parameters
</div>
"""
             st.markdown(con, unsafe_allow_html=True)
        
        elif selected_section == "WasteLand & Forest":
            years = [2012, 2013, 2014, 2015, 2016, 2017, 2018]
            land_cover_types = ["WasteLand", "Deciduous Forest", "Evergreen Forest", "Degraded/Scrub Fores"]

            # Generate some example data for demonstration
            data = np.random.rand(len(land_cover_types), len(years))

            st.subheader("Scatter Plot of Various Crop Seasons")

            # Create a new figure for the scatter plot
            fig, ax = plt.subplots(figsize=(10, 6))

            for i, land_cover in enumerate(land_cover_types):
                ax.scatter(years, data[i, :], label=land_cover, alpha=0.7)
                ax.plot(years, data[i, :], alpha=0.5)  # Add lightly visible lines

            # Set the x-axis label
            ax.set_xlabel('Year')

            # Set the y-axis label
            ax.set_ylabel('Numeric Value (e.g., Land Cover Area)')

            # Set the title of the plot
            ax.set_title('Scatter Plot of Various Crop Seasons')

            # Add legend
            ax.legend()

            # Display the plot using st.pyplot()
            st.pyplot(fig)

            content = """
<div style="text-align: justify">
<h3>Conclusion</h3>

- When Decidous forest decomposes, it will create Degraded/Scrub Forest, But the Process of Decomposing is a long process, that's why if a forest starts to decompose, it will be considered as a degraded/scrub Forest and decidous Forest will be captured less, that's why we have Positive correlation

- Evergreen has very low positive correlation with wasteland, On which we can only say that if evergreen forest increases the wasteland should be decreased

- The Evergreen Forest shows High Correlation with GDP, but we cant ascertain the reason behind it, as UP is Agro-Based Economy

<h4>NOTE</h4>

The Percentage increase of all these Parameters are quite less i.e lower than 0.8% throught out 7 Years.
</div>
"""
            st.markdown(content, unsafe_allow_html=True)
        elif selected_section == "Grasslands Over Years":
            years = [2012, 2013, 2014, 2015, 2016, 2017, 2018]

            # Example data for demonstration (replace this with your actual data)
            # Ensure that df is a valid DataFrame containing the 'Grassland' column
            grassland_data = df['Grassland'].values  # Example data size for Grassland

            st.subheader("Bar Chart for Grassland Over Years")

            # Create a new figure for the bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(years, grassland_data, alpha=0.7, color='lightgreen')

            # Set the x-axis label
            ax.set_xlabel('Year')

            # Set the y-axis label
            ax.set_ylabel('Grassland Size')

            # Set the title of the plot
            ax.set_title('Bar Chart for Grassland Over Years')

            # Display the plot using st.pyplot()
            st.pyplot(fig)


        elif selected_section == "crop seasons":
            years = [2012, 2013, 2014, 2015, 2016, 2017, 2018]
            land_cover_types = ["Rabi Crop", "Zaid Crop", "Kharif Crop", "Current Fallow"]

            # Generate some example data for demonstration
            data = np.random.rand(len(land_cover_types), len(years))

            st.subheader("Scatter Plot of Various Crop Seasons")

            # Create a new figure for the scatter plot
            fig, ax = plt.subplots(figsize=(10, 6))

            for i, land_cover in enumerate(land_cover_types):
                ax.scatter(years, data[i, :], label=land_cover, alpha=0.7)
                ax.plot(years, data[i, :], alpha=0.5)  # Add lightly visible lines

            # Set the x-axis label
            ax.set_xlabel('Year')

            # Set the y-axis label
            ax.set_ylabel('Numeric Value (e.g., Land Cover Area)')

            # Set the title of the plot
            ax.set_title('Scatter Plot of Various Crop Seasons')

            # Add legend
            ax.legend()

            # Display the plot using st.pyplot()
            st.pyplot(fig)
            content="""
<div style="text-align: justify">
<h3>Conclusion</h3>

- For the Current Fallow we can see, whenever it is increasing, the other Crop Season Area Percentage is Decreasing

- But, On a closer look, you can see the current fallow is happening only for Rabi and Zaid Crop, which means in the Kharif Season, People are farming the crops

</div>

"""

        elif selected_section == "Built-up":
            years = [2012, 2013, 2014, 2015, 2016, 2017, 2018]

            # Example data for demonstration (replace this with your actual data)
            # Ensure that df is a valid DataFrame containing the 'Built-up' column
            built_up = df['Built-up'].values  # Example data size for Built-up

            st.subheader("Bar Chart for Built-up Over Years")

            # Create a new figure for the bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(years, built_up, alpha=0.7, color='lightgreen')

            # Set the x-axis label
            ax.set_xlabel('Year')

            # Set the y-axis label
            ax.set_ylabel('Built-up Size')

            # Set the title of the plot
            ax.set_title('Bar Chart for Built-up Over Years')

            # Display the plot using st.pyplot()
            st.pyplot(fig)


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
    
