import streamlit as st
import pandas as pd
import numpy as np

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="MomScience Calculator", layout="wide") 


# --- 2. THE TOP CONTENT ---

st.title("MomScience: False Discovery Rate Calculator")
    
# Descriptive text
st.write("""
This tool calculates the **False Discovery Rate (FDR)** along with **False Omission Rate (FOR)** using Bayes' Theorem.
""")
st.markdown("---")


# ==============================================================================
# --- 3. SIDEBAR: Left Panel (Inputs & Navigation) ---
# ==============================================================================

with st.sidebar:
    # --- Input Parameters ---
    st.header("⚙️ Input Parameters (%)")

    # Single Percentage Input for Prevalence
    prevalence_percent_input = st.number_input(
        "Prevalence (%)",
        min_value=0.01, max_value=100.0, value=10.0, step=1.0, format="%.2f",
        key="side_prevalence",
        help="e.g., Enter 10.0 for 10% prevalence meaning that 10% of population is affected by the condition."
    )

    sensitivity_percent_input = st.number_input(
        "Sensitivity: (%)",
        min_value=0.0, max_value=100.0, value=95.0, step=1.0, format="%.2f",
        key="side_sensitivity",
        help="e.g., Enter 95.0 for 95% sensitivity meaning that the test correctly identifies 95% of people who have the condition."
    )

    specificity_percent_input = st.number_input(
        "Specificity (%)",
        min_value=0.0, max_value=100.0, value=97.0, step=1.0, format="%.2f",
        key="side_specificity",
        help="e.g., Enter 97.0 for 97% specificity meaning that the test correctly identifies 97% of people who do not have the condition."
    )

    st.markdown("---")

    st.header("🔍 Choose Section")
    selected_section = st.radio(
        "Navigate the calculator:",
        [
            "Overview",
            "Rates for Given Inputs",
            "Rates with Multiple Prevalences",
            "Inverse Calculator: from False Discovery Rate to Prevalence"
        ],
        index=0, # Default to "Overview"
    )
    
    st.markdown("---")
    st.markdown("by ScientistMom")
    

# ==============================================================================
# --- CORE CALCULATIONS ---
# ==============================================================================

# Convert percentage inputs back to proportion (0 to 1) for calculation
prevalence = prevalence_percent_input / 100.0
sensitivity = sensitivity_percent_input / 100.0
specificity = specificity_percent_input / 100.0
prevalence_display = f"{prevalence_percent_input:.2f}%" # Display format is just the percentage

# Ensure inputs are valid before proceeding
if not (0.0001 <= prevalence <= 1.0 and 0 <= sensitivity <= 1 and 0 <= specificity <= 1):
    st.error("Please ensure all input percentages are valid (0.01-100).")
    st.stop() 

# 1. Calculate probabilities (fractions of the total population)
prob_true_positive = sensitivity * prevalence
prob_false_positive = (1 - specificity) * (1 - prevalence)
prob_false_negative = (1 - sensitivity) * prevalence
prob_true_negative = specificity * (1 - prevalence)

# 2. P(Positive Test) and P(Negative Test)
prob_positive_test = prob_true_positive + prob_false_positive
prob_negative_test = prob_true_negative + prob_false_negative

# Calculate FDR and FOR (needed for all sections)
if prob_positive_test > 0 and prob_negative_test > 0:
    fdr = prob_false_positive / prob_positive_test
    FOR = prob_false_negative / prob_negative_test
else:
    fdr = np.nan
    FOR = np.nan

# ==============================================================================
# --- 4. MAIN CONTENT SECTIONS (Right Side Display) ---
# ==============================================================================

if selected_section == "Overview":
    st.header("Baseline Population Breakdown")

    # Calculate the percentage widths for the bar
    prevalence_percent_calc = prevalence * 100
    non_prevalence_percent = 100.0 - prevalence_percent_calc

    # Display percentages above the bar
    text_col_left_1, text_col_right_1 = st.columns([1, 4])
    with text_col_left_1:
        st.markdown(f'<p style="color:#F44336; font-weight:bold; margin-bottom: 0px;">{prevalence_percent_calc:.2f}%</p>', unsafe_allow_html=True)
    with text_col_right_1:
        st.markdown(f'<p style="color:#2196F3; font-weight:bold; text-align: right; margin-bottom: 0px;">{non_prevalence_percent:.2f}%</p>', unsafe_allow_html=True)

    # HTML for the segmented bar (solid red/blue) - HEIGHT REDUCED TO 50px
    html_code_1 = f"""
    <div style="
        display: flex;
        height: 50px;
        width: 100%;
        border: 1px solid #333;
        border-radius: 5px;
        overflow: hidden;
        margin-top: 5px;
    ">
        <div style="
            background-color: #F44336; /* Solid Red: Prevalence */
            width: {prevalence_percent_calc}%;
            height: 100%;
        ">
        </div>
        <div style="
            background-color: #2196F3; /* Solid Blue: 1-Prevalence */
            width: {non_prevalence_percent}%;
            height: 100%;
        ">
        </div>
    </div>
    """
    st.markdown(html_code_1, unsafe_allow_html=True)
    st.caption(f"""
        **Red Section:** Population **with** the condition (Prevalence: {prevalence_display}).
        **Blue Section:** Population **without** the condition (1 - Prevalence).
    """)

    st.divider()

    st.subheader("Test Characteristics")
    st.markdown(f"""
    * **Sensitivity:** The test correctly identifies **{sensitivity_percent_input:.2f}%** of people **with** the condition.
    * **Specificity:** The test correctly identifies **{specificity_percent_input:.2f}%** of people **without** the condition.
    """)

elif selected_section == "Rates for Given Inputs":
    st.header("Predictive Rate Results")

    if not np.isnan(fdr):

        # --- Pop-up Help Descriptions ---
        fdr_help_desc = """
        **False Discovery Rate (FDR)**: The risk that a **Positive** test result is **wrong**.
        It is the probability that a person who tested positive does *not* actually have the condition.

        Formula: P(No Condition | Test Positive) = FP / (TP + FP)
        """

        for_help_desc = """
        **False Omission Rate (FOR)**: The risk that a **Negative** test result is **wrong**.
        It is the probability that a person who tested negative *does* actually have the condition.

        Formula: P(Condition | Test Negative) = FN / (FN + TN)
        """

        col_fdr, col_for = st.columns(2)

        with col_fdr:
            st.metric(
                label="False Discovery Rate (FDR)",
                value=f"{fdr * 100:.2f} %",
                help=fdr_help_desc # Pop-up explanation
            )

        with col_for:
            st.metric(
                label="False Omission Rate (FOR)",
                value=f"{FOR * 100:.2f} %",
                help=for_help_desc # Pop-up explanation
            )

        # Kept the summary interpretation below the metrics
        st.success(f"""
        **FDR Interpretation:** If a person receives a **positive test** result, there is a
        **{fdr * 100:.2f}% chance** that they **do not actually have the condition** (False Positive).
        """)

        st.info(f"""
        **FOR Interpretation:** If a person receives a **negative test** result, there is a
        **{FOR * 100:.2f}% chance** that they **do have the condition** (False Negative).
        """)

    else:
        st.warning("Cannot calculate FDR/FOR: One or both probabilities of a test result (positive/negative) are zero based on your inputs.")


elif selected_section == "Rates with Multiple Prevalences":
    st.header("Comparative Analysis by Prevalence")

    # Display current Sensitivity and Specificity
    st.markdown(f"""
    The following rates are calculated using the **current test characteristics**:
    * **Sensitivity:** **{sensitivity_percent_input:.2f}%**
    * **Specificity:** **{specificity_percent_input:.2f}%**
    """)
    st.markdown("---") # Visual separator between inputs and table

    # Define the target prevalences in proportional and human-readable form
    prevalence_data = [
        {"Label": "1 in 2 (50%)", "P": 0.5},
        {"Label": "1 in 10 (10%)", "P": 0.1},
        {"Label": "1 in 100 (1%)", "P": 0.01},
        {"Label": "1 in 1,000 (0.1%)", "P": 0.001},
        {"Label": "1 in 10,000 (0.01%)", "P": 0.0001},
    ]

    results = []

    # Function to calculate all metrics for a given prevalence (P)
    def calculate_metrics_for_table(P, Se, Sp):
        # Fractions of Total Population
        TP = Se * P
        FP = (1 - Sp) * (1 - P)
        FN = (1 - Se) * P
        TN = Sp * (1 - P)

        P_Test_Positive = TP + FP
        P_Test_Negative = FN + TN

        # FDR (False Discovery Rate)
        FDR_val = FP / P_Test_Positive if P_Test_Positive > 0 else 0

        # FOR (False Omission Rate)
        FOR_val = FN / P_Test_Negative if P_Test_Negative > 0 else 0

        return FDR_val, FOR_val

    # Populate results list
    for item in prevalence_data:
        P = item["P"]
        FDR_val, FOR_val = calculate_metrics_for_table(P, sensitivity, specificity)

        results.append({
            "Prevalence (P)": item["Label"],
            "FDR (False Discovery Rate)": f"{FDR_val * 100:.2f}%",
            "FOR (False Omission Rate)": f"{FOR_val * 100:.2f}%"
        })

    # Create and display the DataFrame
    df = pd.DataFrame(results)
    st.table(df)


elif selected_section == "Inverse Calculator: from False Discovery Rate to Prevalence":
    st.header("Inverse Calculator: Find Required Prevalence")
    st.markdown("Calculate the required **Prevalence** to achieve a target False Discovery Rate (FDR) based on the current **Sensitivity** and **Specificity** from the sidebar.")

    # --- Use inputs from the sidebar ---
    Se_calc = sensitivity # Already calculated from sidebar input
    Sp_calc = specificity # Already calculated from sidebar input
    
    st.subheader("Current Test Characteristics (from Sidebar):")
    st.markdown(f"""
    * **Sensitivity:** **{sensitivity_percent_input:.2f}%**
    * **Specificity:** **{specificity_percent_input:.2f}%**
    """)
    
    # --- Single Input for the Target FDR ---
    rate_calc = st.number_input(
        "Target False Discovery Rate (FDR) (%)",
        min_value=0.0, max_value=100.0, value=5.0, step=1.0, format="%.2f",
        key="FDR_calc_input",
        help="Enter the maximum acceptable False Discovery Rate you want to achieve."
    ) / 100.0
    
    # --- Calculation Logic ---

    calculated_result = None # Stored as a proportion (0 to 1)
    
    # Target Metric (Prevalence P) and Known Rate (FDR R) are fixed for this section
    R = rate_calc
    Se = Se_calc
    Sp = Sp_calc

    # Formula: P = [(1 - Sp) * (1 - FDR)] / [(FDR * Se) - (FDR * (1 - Sp)) + (1 - Sp)]

    term_one_minus_Sp = (1 - Sp)

    numerator = term_one_minus_Sp * (1 - R)
    denominator = (R * Se) - (R * term_one_minus_Sp) + term_one_minus_Sp

    if denominator == 0:
        if R == 0 and term_one_minus_Sp == 0:
            st.info("To achieve 0% FDR when Specificity is 100%, Prevalence can be anything (as long as it's not 0).")
            calculated_result = None
        else:
            st.error("Invalid input combination: Denominator is zero. Check if your Specificity/Sensitivity are realistic for your target FDR.")
            calculated_result = None
    elif numerator == 0:
        calculated_result = 0.0
    else:
        P = numerator / denominator
        calculated_result = P

    # --- Display Results ---
    if calculated_result is not None:
        calculated_percent = calculated_result * 100

        if calculated_percent < 0 or calculated_percent > 100:
            error_message = f"The calculated Prevalence required is **{calculated_percent:.2f}%**. This is **impossible** (must be between 0% and 100%). Adjust your target FDR or the test characteristics."
            st.error(error_message)
        elif calculated_result == 0.0:
            # Using LaTeX for the infinity symbol
            st.success(f"Required **Prevalence (P)** to achieve {rate_calc*100:.2f}% FDR: **0.00%** (1 in $\\infty$)")
        else:
            one_in_x = 1 / calculated_result
            st.success(f"Required **Prevalence (P)** to achieve {rate_calc*100:.2f}% FDR: **{calculated_percent:.2f}%** (1 in **{one_in_x:.2f}**)")
    st.divider()
