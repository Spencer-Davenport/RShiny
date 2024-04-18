#library shiny package
library(png)
library(shiny)
library(mice)
library(randomForest)
library(readr)
library(readxl)
library(DT)

#read in data used
data <- read_csv("data/data.csv")

#read in final model
final_model <- readRDS("models/Model.rds")

# Read the image file
file_con <- file("plots/rf.png", "rb")

image_data <- readBin(file_con, "raw", file.info("plots/rf.png")$size)

close(file_con)

# Encode the image data to base64
encoded_image <- base64enc::base64encode(image_data)

# Define UI with CSS styling
ui <- fluidPage(

  # Title panel
  titlePanel("ICU Mortality Prediction"),

  # Custom CSS styles
  tags$head(
    tags$style(HTML("
      .tab-content {
        padding: 20px;
      }
      .important-warning {
        font-weight: bold;
        color: #003594;
      }
      .info-box {
        background-color: #f0f0f0;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
        color: #003594;
      }
      .data-box {
        background-color: #f0f0f0;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
        color: #003594;
      }
      .model-box {
        background-color: #f0f0f0;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
        color: #003594;
      }
    "))
  ),

  # Tabset panel with multiple tabs
  tabsetPanel(

    # Home tab
    tabPanel("Home",
             align = "center",
             # Welcome message and warning
             div(class = "info-box",
                 tags$h3("Welcome to the ICU Patient Mortality Prediction app!", style = "font-weight: bold;"),
                 tags$p("Created by: Madeline Peterson, Nina Gakii, and Spencer Reed Davenport",
                        style = "font-style: italic;"),
                 tags$p("For BIOST 2094 Advanced R", style = "font-style: italic; font-size: 80%;"),
                 tags$p("This app is designed to predict ICU patient mortality based on patient data.*" ,
                        style = "font-weight: bold;"),
                 tags$p("In the ICU Mortality Prediction app, we utilize advanced methods to forecast
                        mortality rates among Intensive Care Unit (ICU) patients. This tool is pivotal
                        for comparing treatment efficacy across diverse patient populations, enabling
                        researchers to account for differences in illness severity, age, and other
                        factors. Unlike traditional methods, our approach focuses on individual patient
                        prognosis, leveraging data from the first two days of ICU admission. By
                        providing personalized predictions, we empower clinicians to make informed
                        decisions, ultimately enhancing patient care and outcomes."),
                 tags$br(),
                 tags$br(),
                 tags$br(),
                 tags$h3("Important Warning:", class = "important-warning"),
                 tags$p("While this app provides predictions, it's crucial to understand its limitations.
                        ICU mortality prediction is a complex task affected by numerous factors.
                        This app uses a simplified model and should not be relied upon for clinical
                        decision-making."),
                 tags$br(),
                 tags$br(),
                 tags$br(),
                 tags$p("*Please note that the predictions made by this app should not be used for
                 clinical decisions.", style = "font-style: italic; font-size:70%;")
             )
    ),

    # Data tab
    tabPanel("Data",
             div(class = "data-box",
                 # Introduction to the data used
                 tags$p("In this section, we introduce the data used for ICU mortality prediction."),
                 tags$p("The dataset utilized in this ICU Mortality Prediction app comprises a comprehensive
                    array of variables collected during the first 48 hours of patients' stay
                    in the Intensive Care Unit (ICU). These variables encompass a wide range of
                    demographic information, vital signs, laboratory results, and clinical parameters.
                    By leveraging this rich dataset, our predictive model gains insights into the complex
                    interplay of factors influencing patient outcomes."),
                 tags$p("Below is a table of all variables and their descriptions:")),
             DTOutput("excel_table")
    ),

    # Model tab
    tabPanel("Model",
             div(class = "model-box",
                 # Introduction to the modeling technique
                 tags$p("In this section, we discuss the modeling technique used for ICU mortality prediction."),
                 tags$p("We employ a Random Forest algorithm, which is a powerful ensemble learning method
                    renowned for its ease of use and robust performance across various tasks.
                    It belongs to the class of ensemble methods, constructing an ensemble of decision
                    trees to make predictions. The 'forest' it creates consists of multiple decision
                    trees, typically trained using the bagging method, where each tree is trained on
                    a random subset of the training data. This introduction of randomness during tree
                    construction promotes diversity among the trees, leading to more accurate and
                    stable predictions.")),
             tags$br(), # Line break
             # Image illustrating Random Forests
             tags$img(src = paste0("data:image/png;base64,", encoded_image)),
             tags$br(),
             div(class = "model-box",
                 tags$p("One of the key advantages of Random Forest is its versatility.
                    It can handle both classification and regression tasks, making it applicable
                    to a wide range of real-world problems. Understanding and tuning the hyperparameters
                    of Random Forest is straightforward, as there are only a few parameters to consider.
                    Additionally, Random Forest is robust to overfitting, especially with a sufficient
                    number of trees, which is a common issue in machine learning. Furthermore,
                    Random Forest provides a measure of feature importance, allowing users to identify
                    the most influential features in the prediction process."),
                 tags$p("However, there are also some limitations to consider. Achieving higher accuracy
                    with Random Forest often requires a larger number of trees, which can increase
                    the complexity of the model and slow down prediction speed, particularly for
                    real-time applications. Moreover, while Random Forest excels at prediction,
                    it does not provide insights into the underlying relationships within the data,
                    limiting its interpretability."),
                 tags$p("Despite these limitations, Random Forest remains a popular choice in machine
                    learning due to its excellent performance, ease of use, and ability to handle
                    complex datasets. It serves as a reliable and versatile tool for data scientists
                    and practitioners across various domains."))
    ),

    # Our model tab
    tabPanel("Our Model",
             div(class = "model-box",
                 # Introduction to the modeling technique
                 tags$p("In this section, we will discuss the model that is implemented in this shiny App."),
                 tags$p("We employed a random forest model using the 48-hour ICU data sourced from the
                        2012 PhysioNet Challenge. To facilitate model training, we summarized the
                        time series variables by computing their medians. Addressing missing values,
                        we utilized the missRanger package in R, leveraging its random forest-based
                        imputation method for variables with less than 40% missing data. Variables
                        exceeding this threshold were omitted from the analysis. To track imputations,
                        we introduced indicator variables (0: not missing, 1: missing). The model
                        fitting process was conducted with the caret package in R, renowned for its
                        robust machine learning capabilities. We adopted 10-fold cross-validation and
                        allocated 70% of the dataset for training purposes."),
             tags$p("After finalizing the model, we conducted a comprehensive analysis to identify
             the most influential variables. Using the concept of feature importance, we identified
             the top 10 variables that significantly contributed to the model's predictive power."),
             tags$p("Furthermore, we employed Principal Component Analysis (PCA) to delve deeper
             into the data structure and discern which variables held the most explanatory power.
             PCA is a dimensionality reduction technique that transforms a set of correlated variables
             into a smaller set of uncorrelated variables known as principal components. These
             components are ordered by the amount of variance they explain in the data."),
             tags$p("In essence, PCA works by computing the eigenvectors and eigenvalues of the covariance
             matrix of the data. The eigenvectors represent the directions of maximum variance in
             the data, while the eigenvalues indicate the magnitude of variance along these directions.
             The principal components are then derived from these eigenvectors, with each component
             capturing a different proportion of the total variance in the dataset."),
             tags$p("By performing PCA, we sought to compare the model's performance using the top
             10 variables with its performance when considering a reduced set of variables derived
             from PCA. Specifically, we evaluated the model fit based on the variables included in
             the first principal component, which encapsulates the maximum variance in the data.
             This comparison allowed us to assess whether the information captured by the top 10
             variables aligns with the overarching patterns revealed by PCA."),
             tags$p("Ultimately, after thorough evaluation, we opted to proceed with the model fit
                    using the top 10 variables. This decision was informed by a comprehensive analysis
                    of both feature importance and the explanatory power of variables derived from PCA."),
            tags$p("After conducting hyperparameter tuning, focusing on optimizing
            the mtry hyperparameter, through 10-fold cross-validation, we proceeded to train
            a final model on the entire dataset."),
            tags$p("In random forest models, mtry represents the number of variables randomly
            sampled at each split when constructing each tree in the forest. This hyperparameter
            governs the level of randomness in feature selection, influencing the model's ability
            to generalize to unseen data. By fine-tuning mtry, we aimed to strike a balance between
            model complexity and predictive performance."),
            tags$p("Hyperparameter tuning involves systematically searching for the optimal
            values of model parameters that maximize performance metrics such as accuracy,
            sensitivity, specificity, and area under the curve (AUC). This process is crucial
            for optimizing model performance and mitigating overfitting."),
            tags$p("Following hyperparameter tuning, we evaluated the final model's performance
            across various metrics:"),
            tags$br(),
            verbatimTextOutput("performance_metrics"),
            tags$br(),
            tags$p("These metrics provide insights into the model's overall effectiveness in
                   classifying instances, detecting true positives, and avoiding false positives."),
            tags$p("It's worth noting that while the final model demonstrated commendable accuracy
            and specificity, it exhibited lower sensitivity, indicating potential limitations in
            correctly identifying positive cases. This underscores the importance of considering
            the model's performance characteristics in context, particularly in critical
            decision-making scenarios.Furthermore, we emphasize that the calculator derived from
                   this model should not be used as a standalone tool for making clinical decisions.
                   Rather, its purpose primarily revolves around showcasing the functionalities
                   of the developed shiny App. Our primary objective in this analysis was to gain
                   familiarity with the process of creating a shiny App and demonstrating its
                   capabilities."))
    ),

    tabPanel("Calculator",
             sidebarLayout(
               sidebarPanel(
                 tags$p("Please input values of the variables below. Input multiple values separated
                        by comas. If the variable is missing, please do not input anything in the
                        text box and select the missing box so it is checked."),

                 # Input elements for the calculator
                 textAreaInput("BUN", "Enter Values of Blood urea nitrogen (mg/dL) (Normal: 6 to 24 mg/dL):", value = NULL),
                 checkboxInput("BUN_missing", "Is BUN Missing?", value = FALSE),

                 # Input for median value of WBC
                 textAreaInput("WBC", "Enter Values of White blood cell count (cells/nL) (Normal: 4.5 to 11 cells/nL):", value = NULL),
                 # Checkbox for missing indicator
                 checkboxInput("WBC_missing", "Is WBC Missing?", value = FALSE),

                 # Input for median value of Platelets
                 textAreaInput("Platelets", "Enter Values of Platelets (cells/nL) (Normal: 150 to 400 cells/nL):", value = NULL),
                 # Checkbox for missing indicator
                 checkboxInput("Platelets_missing", "Is Platelets Missing?", value = FALSE),

                 # Input for median value of PaO2
                 textAreaInput("PaO2", "Enter Values of Partial pressure of arterial O2 (mmHg) (Normal: 75 to 100 mmHg):", value = NULL),
                 # Checkbox for missing indicator
                 checkboxInput("PaO2_missing", "Is PaO2 Missing?", value = FALSE),

                 # Input for median value of PaCO2
                 textAreaInput("PaCO2", "Enter Values of partial pressure of arterial CO2 (mmHg) (Normal: 35 to 45 mmHg):", value = NULL),
                 # Checkbox for missing indicator
                 checkboxInput("PaCO2_missing", "Is PaCO2 Missing?", value = FALSE),

                 # Input for median value of SysABP
                 textAreaInput("SysABP", "Enter Values of Invasive systolic arterial blood pressure (mmHg) (Normal: 120 to 129 mmHg):", value = NULL),
                 # Checkbox for missing indicator
                 checkboxInput("SysABP_missing", "Is SysABP Missing?", value = FALSE),

                 # Input for median value of GCS
                 textAreaInput("GCS", "Enter Values of Glasgow Coma Score (Range: 3 to 15):", value = NULL),
                 # Checkbox for missing indicator
                 checkboxInput("GCS_missing", "Is GCS Missing?", value = FALSE),

                 # Input for median value of glucose
                 textAreaInput("Glucose", "Enter Values of Serum glucose (mg/dL) (Normal: 70 to 100 mg/dL):", value = NULL),
                 # Checkbox for missing indicator
                 checkboxInput("Glucose_missing", "Is Glucose Missing?", value = FALSE),

                 # Input for median value of Urine
                 textAreaInput("Urine", "Enter Values of Urine output (mL) (Normal: 800 to 2000 mL):", value = NULL),
                 # Checkbox for missing indicator
                 checkboxInput("Urine_missing", "Is Urine Missing?", value = FALSE),

                 # Input for median value of Age
                 sliderInput("Age", "Select age:", min=15, max=90, value=40),


                 #predict button
                 actionButton("predict_button", "Predict")
               ),

               mainPanel(
                 h4("Prediction Result:"),
                 textOutput("prediction_result")
               )
             )
    )
  )
)

# Function to preprocess input data and make predictions
predict_rf <- function(input_data) {
  # Convert numeric inputs to numeric format and handle missing values
  numeric_columns <- c("BUN_median", "WBC_median","Platelets_median",
                       "GCS_median", "SysABP_median", "Glucose_median", "Urine_median",
                       "Age", "PaO2_median", "PaCO2_median")

  for (col in numeric_columns) {
    # Handle missing values
    input_data[[col]][is.na(input_data[[col]])] <- median(data[[col]], na.rm = TRUE)
  }

  # Make predictions using the Random Forest model
  predictions <- predict(final_model, newdata = input_data, type = "prob")

  # Get the predicted class
  predicted_class <- colnames(predictions)[which.max(predictions)]

  # Get the predicted probability of the predicted class
  predicted_probability <- max(predictions)

  return(list(predicted_class = predicted_class, predicted_probability = predicted_probability*100))
}


# Define server logic for the calculator
calculator_server <- function(input, output, session) {
  output$excel_table <- renderDT({
    # Define the path to the Excel file
    excel_file <- "tables/table.xlsx"

    # Read the Excel file
    excel_data <- read_xlsx(excel_file)

    # Use datatable() function from DT package
    datatable(excel_data, options = list(
      pageLength = 43,
      rowCallback = JS(
        'function(row, data) {
          var colorRows = ["Age", "BUN", "GCS", "Glucose", "PaCO2", "PaO2", "Platelets", "SysABP", "Urine", "WBC"];
          var colorRow = ["mortality"];

          if (colorRows.includes(data[1])) {
            $(row).css("background-color", "#8FBC8F");
          }
          if (colorRow.includes(data[1])) {
            $(row).css("background-color", "#DC143C");
          }
        }'
      )
    ))
  })

  #performance metrics table
  output$performance_metrics <- renderPrint({
    performance_text <- "
    - Accuracy: 0.870
    - Sensitivity: 0.987
    - Specificity: 0.139
    - Positive Predictive Value (PPV): 0.877
    - Negative Predictive Value (NPV): 0.639
    - Area Under the Curve (AUC): 0.815
    "
    cat(performance_text)
  })

  observeEvent(input$predict_button, {
    # Prepare input data for prediction

    BUN_values <- unlist(lapply(strsplit(input$BUN, ","), as.numeric))
    bun_median <- ifelse(input$BUN_missing, NA, ifelse(length(BUN_values) == 0, NA, median(BUN_values, na.rm = TRUE)))

    WBC_values <- unlist(lapply(strsplit(input$WBC, ","), as.numeric))
    wbc_median <- ifelse(input$WBC_missing, NA, ifelse(length(WBC_values) == 0, NA, median(WBC_values, na.rm = TRUE)))

    Platelets_values <- unlist(lapply(strsplit(input$Platelets, ","), as.numeric))
    platelets_median <- ifelse(input$Platelets_missing, NA, ifelse(length(Platelets_values) == 0, NA, median(Platelets_values, na.rm = TRUE)))

    gcs_values <- unlist(lapply(strsplit(input$GCS, ","), as.numeric))
    gcs_median <- ifelse(input$GCS_missing, NA, ifelse(length(gcs_values) == 0, NA, median(gcs_values, na.rm = TRUE)))

    SysABP_values <- unlist(lapply(strsplit(input$SysABP, ","), as.numeric))
    sysABP_median <- ifelse(input$SysABP_missing, NA, ifelse(length(SysABP_values) == 0, NA, median(SysABP_values, na.rm = TRUE)))

    Glucose_values <- unlist(lapply(strsplit(input$Glucose, ","), as.numeric))
    glucose_median <- ifelse(input$Glucose_missing, NA, ifelse(length(Glucose_values) == 0, NA, median(Glucose_values, na.rm = TRUE)))


    Urine_values <- unlist(lapply(strsplit(input$Urine, ","), as.numeric))
    urine_median <- ifelse(input$Urine_missing, NA, ifelse(length(Urine_values) == 0, NA, median(Urine_values, na.rm = TRUE)))


    PaO2_values <- unlist(lapply(strsplit(input$PaO2, ","), as.numeric))
    paO2_median <- ifelse(input$PaO2_missing, NA, ifelse(length(PaO2_values) == 0, NA, median(PaO2_values, na.rm = TRUE)))


    PaCO2_values <- unlist(lapply(strsplit(input$PaCO2, ","), as.numeric))
    paCO2_median <- ifelse(input$PaCO2_missing, NA, ifelse(length(PaCO2_values) == 0, NA, median(PaCO2_values, na.rm = TRUE)))


    Age_values <- as.numeric(input$Age)


    input_data <- data.frame(
      BUN_median = bun_median,
      BUN_median_missing = as.numeric(input$BUN_missing),

      WBC_median = wbc_median,
      WBC_median_missing = as.numeric(input$WBC_missing),

      Platelets_median = platelets_median,
      Platelets_median_missing = as.numeric(input$Platelets_missing),

      GCS_median = gcs_median,
      GCS_median_missing = as.numeric(input$GCS_missing),

      SysABP_median = sysABP_median,
      SysABP_median_missing = as.numeric(input$SysABP_missing),

      Glucose_median = glucose_median,
      Glucose_median_missing = as.numeric(input$Glucose_missing),

      Urine_median = urine_median,
      Urine_median_missing = as.numeric(input$Urine_missing),

      Age = Age_values,

      PaO2_median = paO2_median,
      PaO2_median_missing = as.numeric(input$PaO2_missing),

      PaCO2_median = paCO2_median,
      PaCO2_median_missing = as.numeric(input$PaCO2_missing)
    )

    # Make prediction
    prediction <- predict_rf(input_data)

    # Output prediction result
    output$prediction_result <- renderText({
      paste("Predicted Outcome: ", prediction$predicted_class, ", Predicted Probability: ",
            round(prediction$predicted_probability, 2), "%")
    })
  })
}

# Run the application
shinyApp(
  ui = ui,
  server = calculator_server
)
