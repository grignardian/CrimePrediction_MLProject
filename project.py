import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

# 1. Load the datasets
census = pd.read_csv("census.csv")
crime = pd.read_csv("NCRB_2011_Table_1.14.csv", encoding="latin1")

print("===== CENSUS DATA =====")
print(census.head())
print("\n===== CENSUS COLUMNS =====")
print(census.columns)
print("\n============================\n")
print("===== CRIME DATA =====")
print(crime.head())
print("\n===== CRIME COLUMNS =====")
print(crime.columns)

# Rename the state column in crime to match census
crime.rename(columns={"States/ UTs": "State"}, inplace=True)

# 2. Cleanup state names helper
def clean_state(s):
    if not isinstance(s, str):
        return s
    s = s.replace('\xa0', ' ').strip().lower().replace('&', 'and')
    mappings = {
        'aandn islands': 'andaman and nicobar islands',
        'a and n islands': 'andaman and nicobar islands',
        'dandn haveli': 'dadra and nagar haveli',
        'd and n haveli': 'dadra and nagar haveli',
        'delhi ut': 'delhi',
    }
    return mappings.get(s, s)

# Dictionary to fix district name mismatches between crime & census datasets
district_map = {
    # West Bengal
    ("west bengal", "24 parganas north"): "north twenty four parganas",
    ("west bengal", "24 parganas south"): "south twenty four parganas",
    ("west bengal", "north 24 parganas"): "north twenty four parganas",
    ("west bengal", "south 24 parganas"): "south twenty four parganas",
    ("west bengal", "burdwan"): "barddhaman",
    ("west bengal", "coochbehar"): "koch bihar",
    ("west bengal", "hooghly"): "hugli",
    ("west bengal", "howrah"): "haora",
    ("west bengal", "paschim midnapur"): "paschim medinipur",
    ("west bengal", "purab midnapur"): "purba medinipur",
    ("west bengal", "purulia"): "puruliya",
    ("west bengal", "malda"): "maldah",
    
    # Gujarat
    ("gujarat", "ahmedabad"): "ahmadabad",
    ("gujarat", "ahwa-dang"): "the dangs",
    ("gujarat", "dahod"): "dohad",
    ("gujarat", "mehsana"): "mahesana",
    ("gujarat", "kutch"): "kachchh",
    ("gujarat", "kutch (east(g))"): "kachchh",
    ("gujarat", "kutch (west-bhuj)"): "kachchh",
    ("gujarat", "palanpur"): "banaskantha",
    ("gujarat", "himatnagar"): "sabarkantha",
    ("gujarat", "panchmahal"): "panchmahal",
    ("gujarat", "kheda north"): "kheda",
    
    # Maharashtra
    ("maharashtra", "ahmednagar"): "ahmadnagar",
    ("maharashtra", "nasik"): "nashik",
    ("maharashtra", "raigad"): "raigarh",
    ("maharashtra", "beed"): "bid",
    ("maharashtra", "buldhana"): "buldana",
    
    # Karnataka
    ("karnataka", "cbpura"): "chikkaballapura",
    ("karnataka", "chamarajnagar"): "chamarajanagar",
    ("karnataka", "chickmagalur"): "chikmagalur",
    ("karnataka", "dakshin kannada"): "dakshina kannada",
    ("karnataka", "yadgiri"): "yadgir",
    ("karnataka", "mangalore"): "dakshina kannada",
    
    # Bihar
    ("bihar", "purnea"): "purnia",
    ("bihar", "nawadah"): "nawada",
    ("bihar", "bhabhua"): "kaimur",
    ("bihar", "motihari"): "purbi champaran",
    ("bihar", "bettiah"): "pashchim champaran",
    
    # Andhra Pradesh
    ("andhra pradesh", "ranga reddy"): "rangareddy",
    ("andhra pradesh", "mahaboobnagar"): "mahbubnagar",
    ("andhra pradesh", "cuddapah"): "ysr",
    ("andhra pradesh", "nellore"): "sri potti sriramulu nellore",
    ("andhra pradesh", "prakasham"): "prakasam",
    
    # Uttar Pradesh
    ("uttar pradesh", "badaun"): "budaun",
    ("uttar pradesh", "chandoli"): "chandauli",
    ("uttar pradesh", "chitrakoot dham"): "chitrakoot",
    ("uttar pradesh", "j.p.nagar"): "jyotiba phule nagar",
    ("uttar pradesh", "khiri"): "kheri",
    ("uttar pradesh", "kushi nagar"): "kushinagar",
    ("uttar pradesh", "raibareilly"): "rae bareli",
    ("uttar pradesh", "sidharthnagar"): "siddharth nagar",
    ("uttar pradesh", "st.ravidasnagar"): "sant ravidas nagar",
    ("uttar pradesh", "hathras"): "mahamaya nagar",
    ("uttar pradesh", "sant kabirnagar"): "sant kabir nagar",
    ("uttar pradesh", "fatehgarh"): "farrukhabad",
    ("uttar pradesh", "gautambudh nagar"): "gautam buddha nagar",
    
    # Tamil Nadu
    ("tamil nadu", "kanchipuram"): "kancheepuram",
    ("tamil nadu", "villupuram"): "viluppuram",
    ("tamil nadu", "trichy"): "tiruchirappalli",
    ("tamil nadu", "pudukottai"): "pudukkottai",
    ("tamil nadu", "ramnathapuram"): "ramanathapuram",
    ("tamil nadu", "sivagangai"): "sivaganga",
    ("tamil nadu", "thoothugudi"): "thoothukkudi",
    ("tamil nadu", "thirunelveli"): "tirunelveli",
    ("tamil nadu", "nilgiris"): "the nilgiris",
    ("tamil nadu", "thiruvannamalai"): "tiruvannamalai",
    
    # Kerala
    ("kerala", "trivandrum"): "thiruvananthapuram",
    ("kerala", "alapuzha"): "alappuzha",
    ("kerala", "kasargod"): "kasaragod",
    ("kerala", "wayanadu"): "wayanad",
    
    # Orissa
    ("orissa", "balasore"): "baleshwar",
    ("orissa", "nowrangpur"): "nabarangapur",
    ("orissa", "malkangir"): "malkangiri",
    ("orissa", "baragarh"): "bargarh",
    ("orissa", "bolangir"): "balangir",
    ("orissa", "sonepur"): "subarnapur",
    ("orissa", "angul"): "anugul",
    ("orissa", "khurda"): "khordha",
    ("orissa", "keonjhar"): "kendujhar",
    ("orissa", "jagatsinghpur"): "jagatsinghpur",
    ("orissa", "deogarh"): "debagarh",
    
    # Delhi
    ("delhi", "north-west"): "north west delhi",
    ("delhi", "south-west"): "south west delhi",
    ("delhi", "north-east"): "north east delhi",
    ("delhi", "new delhi"): "new delhi",
    ("delhi", "central"): "central delhi",
    ("delhi", "east"): "east delhi",
    ("delhi", "north"): "north delhi",
    ("delhi", "south"): "south delhi",
    ("delhi", "west"): "west delhi",
    
    # Chhattisgarh
    ("chhattisgarh", "dantewara"): "dantewada",
    ("chhattisgarh", "koriya"): "korea",
    ("chhattisgarh", "bizapur"): "bijapur",
    ("chhattisgarh", "sarguja"): "surguja",
    ("chhattisgarh", "janjgir"): "janjgir champa",
    ("chhattisgarh", "jagdalpur"): "bastar",
    
    # Jharkhand
    ("jharkhand", "jamshedpur"): "purbi singhbhum",
    ("jharkhand", "chaibasa"): "pashchimi singhbhum",
    ("jharkhand", "lohardagga"): "lohardaga",
    ("jharkhand", "saraikela"): "saraikela kharsawan",
    ("jharkhand", "sahebganj"): "sahibganj",
    
    # Madhya Pradesh
    ("madhya pradesh", "khargon"): "west nimar",
    ("madhya pradesh", "khandwa"): "east nimar",
    ("madhya pradesh", "ashok nagar"): "ashoknagar",
    ("madhya pradesh", "datiya"): "datia",
    ("madhya pradesh", "narsinghpur"): "narsimhapur",
    ("madhya pradesh", "sihore"): "sehore",
    ("madhya pradesh", "umariya"): "umaria",
    ("madhya pradesh", "chhatarpur"): "chhattarpur",
    
    # Punjab
    ("punjab", "bhatinda"): "bathinda",
    ("punjab", "ferozpur"): "firozpur",
    ("punjab", "ropar"): "rupnagar",
    ("punjab", "sas ngr"): "mohali",
    ("punjab", "sbs nagar"): "shahid bhagat singh nagar",
    ("punjab", "batala"): "gurdaspur",
    ("punjab", "khanna"): "ludhiana",
    
    # Uttarakhand
    ("uttarakhand", "rudra prayag"): "rudraprayag",
    ("uttarakhand", "udhamsingh nagar"): "udham singh nagar",
    
    # Haryana
    ("haryana", "hissar"): "hisar",
    
    # Rajasthan
    ("rajasthan", "jhunjhunu"): "jhunjhunun",
    ("rajasthan", "chittorgarh"): "chittaurgarh",
    ("rajasthan", "jalore"): "jalor",
    
    # Tripura
    ("tripura", "west"): "west tripura",
    ("tripura", "east"): "east tripura",
    ("tripura", "south"): "south tripura",
    ("tripura", "north"): "north tripura",

    # Sikkim
    ("sikkim", "west"): "west sikkim",
    ("sikkim", "east"): "east sikkim",
    ("sikkim", "south"): "south sikkim",
    ("sikkim", "north"): "north sikkim",

    # Arunachal Pradesh
    ("arunachal pradesh", "k/kumey"): "kurung kumey",
    ("arunachal pradesh", "kameng east"): "east kameng",
    ("arunachal pradesh", "kameng west"): "west kameng",
    ("arunachal pradesh", "papum pare"): "papumpare",
    ("arunachal pradesh", "siang east"): "east siang",
    ("arunachal pradesh", "siang upper"): "upper siang",
    ("arunachal pradesh", "siang west"): "west siang",
    ("arunachal pradesh", "subansiri lower"): "lower subansiri",
    ("arunachal pradesh", "subansiri upper"): "upper subansiri",

    # Meghalaya
    ("meghalaya", "garo hills east"): "east garo hills",
    ("meghalaya", "garo hills south"): "south garo hills",
    ("meghalaya", "garo hills west"): "west garo hills",
    ("meghalaya", "khasi hills east"): "east khasi hills",
    ("meghalaya", "khasi hills west"): "west khasi hills",
    ("meghalaya", "ri-bhoi"): "ri bhoi",

    # Jammu & Kashmir
    ("jammu and kashmir", "baramulla"): "baramula",
    ("jammu and kashmir", "budgam"): "badgam",
    ("jammu and kashmir", "poonch"): "punch",
    ("jammu and kashmir", "shopian"): "shupiyan",

    # Himachal Pradesh
    ("himachal pradesh", "lahaul-spiti"): "lahul and spiti",

    # Dadra & Nagar Haveli
    ("dadra and nagar haveli", "d&n haveli"): "dadra and nagar haveli"
}

# Cleanup district names helper
def clean_district(row):
    d = row['District']
    s = row['State_clean']
    if not isinstance(d, str):
        return d
    d = d.replace('\xa0', ' ').strip().lower()
    
    # Exclude totals and railway divisions
    if d in ['total', 'railways', 'g.r.p.', 'g.r.p', 'w.rly ahmedabad', 'w.rly vadodara', 'cid crime', 'grp']:
        return ''
        
    suffixes = [
        ' commr.', ' commr', ' rural', ' rly', ' rly.', ' g.r.p.', ' g.r.p', 
        ' city', ' railway', ' railways', ' dist.', ' district', ' dist',
        ' commissionerate', ' urban', ' suburban', ' city.', ' commr. ', 
        ' dcp bbsr', ' dcp ctc', ' srp(cuttack)', ' srp(rourkela)'
    ]
    for suf in suffixes:
        if d.endswith(suf) or d.startswith(suf):
            d = d.replace(suf, '').strip()
            
    d = d.replace(' dcp ', '').replace(' srp ', '').replace('cp ', '').strip()
    
    # Return matched name from district_map if present
    return district_map.get((s, d), d)

# 3. Clean and merge
census["State_clean"] = census["State"].apply(clean_state)
crime["State_clean"] = crime["State"].apply(clean_state)
census["Dist_clean"] = census.apply(clean_district, axis=1)
crime["Dist_clean"] = crime.apply(clean_district, axis=1)

# Filter out empty/unnecessary records
crime_filtered = crime[crime["Dist_clean"] != ""]
crime_filtered = crime_filtered[~crime_filtered["Dist_clean"].isin(["total", "total (state)", "total (ut)", "total (all-india)"])]

# Sum up crime numbers for the same district (e.g. city + rural parts)
crime_numeric = [col for col in crime_filtered.columns if col not in ["Sr. No.", "State", "District", "State_clean", "Dist_clean"]]
crime_grouped = crime_filtered.groupby(["State_clean", "Dist_clean"])[crime_numeric].sum().reset_index()

merged = pd.merge(
    census,
    crime_grouped,
    left_on=["State_clean", "Dist_clean"],
    right_on=["State_clean", "Dist_clean"],
    how="inner"
)

print("\nMerged rows:", len(merged))
print(merged.head())

# Clean up Growth column formatting
merged["Growth"] = merged["Growth"].str.replace("%", "", regex=False).str.strip().astype(float)
print(merged[["Population", "Growth", "Sex-Ratio", "Literacy"]].head())

# 4. Prepare features and target variables
X = merged[["Population", "Growth", "Sex-Ratio", "Literacy"]]
y = merged["Total Cog. Crime Under IPC"]

print(X.head())
print(y.head())

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# 5. Train models
# Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Random Forest model trained successfully!")

# Multiple Linear Regression model
mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train)
print("Multiple Linear Regression model trained successfully!")

# Make predictions and evaluate Random Forest
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Make predictions and evaluate MLR
mlr_predictions = mlr_model.predict(X_test)
mlr_mae = mean_absolute_error(y_test, mlr_predictions)
mlr_mse = mean_squared_error(y_test, mlr_predictions)
mlr_r2 = r2_score(y_test, mlr_predictions)

print("\n===== RANDOM FOREST PERFORMANCE =====")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

print("\n===== MULTIPLE LINEAR REGRESSION PERFORMANCE =====")
print("Mean Absolute Error (MAE):", mlr_mae)
print("Mean Squared Error (MSE):", mlr_mse)
print("R² Score (R^2):", mlr_r2)
print("Coefficients:", mlr_model.coef_)
print("Intercept:", mlr_model.intercept_)

# Print feature importances
importance = model.feature_importances_
print("\n===== FEATURE IMPORTANCES =====")
for feature, value in zip(X.columns, importance):
    print(f"{feature}: {value:.4f}")

# Plot feature importances
plt.figure(figsize=(8, 5))
plt.bar(X.columns, importance)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
if not st.runtime.exists():
    plt.show()
else:
    plt.close()

# Show comparison table of actual vs predicted
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "RF Predicted": predictions.astype(int),
    "MLR Predicted": mlr_predictions.astype(int)
})
print(comparison.head(10))

# 6. Streamlit User Interface
st.set_page_config(page_title="Crime Prediction System", layout="wide")
st.sidebar.header("Inference Parameters")
st.sidebar.write("Input demographic values using the - / + buttons below:")

population = st.sidebar.number_input("Population", min_value=1000, max_value=15000000, value=1000000, step=50000)
growth = st.sidebar.number_input("Population Growth Rate (%)", min_value=-20.0, max_value=60.0, value=15.0, step=0.5)
sex_ratio = st.sidebar.number_input("Sex Ratio (Females per 1000 Males)", min_value=700, max_value=1200, value=940, step=5)
literacy = st.sidebar.number_input("Literacy Rate (%)", min_value=30.0, max_value=100.0, value=75.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Selection")
selected_model_name = st.sidebar.selectbox("Select Prediction Model:", ["Random Forest", "Multiple Linear Regression"])

st.sidebar.markdown("---")
st.sidebar.subheader("About the Project")
st.sidebar.info(
    "This system compares a Random Forest Regressor and a Multiple Linear Regression (MLR) model "
    "trained on 2011 India Census data and NCRB crime statistics (618 merged districts)."
)

st.title("District Crime Prediction System")
st.write("Predict the total volume of cognizable crimes (IPC) using district census demographics.")

tab1, tab2, tab3 = st.tabs(["Predict Crimes", "Model Performance", "Data Overview"])

with tab1:
    st.header("Predict Crimes for a District")
    st.write("Adjust the values in the sidebar to calculate the predicted crime volume:")
    
    st.write("### Selected Demographic Inputs:")
    col_v1, col_v2, col_v3, col_v4 = st.columns(4)
    col_v1.metric("Population", f"{population:,}")
    col_v2.metric("Growth Rate", f"{growth}%")
    col_v3.metric("Sex Ratio", f"{sex_ratio}")
    col_v4.metric("Literacy", f"{literacy}%")
    
    st.write(f"Active Prediction Model: **{selected_model_name}**")
    if st.button("Predict"):
        input_df = pd.DataFrame([[population, growth, sex_ratio, literacy]], 
                                columns=["Population", "Growth", "Sex-Ratio", "Literacy"])
        if selected_model_name == "Random Forest":
            prediction = model.predict(input_df)[0]
        else:
            prediction = mlr_model.predict(input_df)[0]
        st.metric(label=f"Predicted Total IPC Crimes ({selected_model_name})", value=f"{int(prediction):,}")

with tab2:
    st.header("Model Performance & Comparison")
    
    col_rf, col_mlr = st.columns(2)
    
    with col_rf:
        st.subheader("Random Forest Regressor")
        st.metric(label="R² Score", value=f"{r2:.4f}")
        st.metric(label="Mean Absolute Error (MAE)", value=f"{int(mae):,}")
        st.metric(label="Mean Squared Error (MSE)", value=f"{int(mse):,}")
        st.metric(label="Intercept", value="—")
        
        st.write("**Feature Importances:**")
        feat_importance = pd.DataFrame({
            'Feature': ["Population", "Growth", "Sex-Ratio", "Literacy"],
            'Importance': model.feature_importances_
        })
        st.dataframe(feat_importance.set_index('Feature'))
        
    with col_mlr:
        st.subheader("Multiple Linear Regression")
        st.metric(label="R² Score", value=f"{mlr_r2:.4f}")
        st.metric(label="Mean Absolute Error (MAE)", value=f"{int(mlr_mae):,}")
        st.metric(label="Mean Squared Error (MSE)", value=f"{int(mlr_mse):,}")
        st.metric(label="Intercept", value=f"{mlr_model.intercept_:.2f}")
        
        st.write("**Coefficients:**")
        coef_df = pd.DataFrame({
            'Feature': ["Population", "Growth", "Sex-Ratio", "Literacy"],
            'Coefficient': mlr_model.coef_
        })
        st.dataframe(coef_df.set_index('Feature'))
        
    st.markdown("---")
    st.subheader("Demographic Trends vs Crimes")
    st.write("Explore how different census factors relate to total cognizable crimes across all 618 merged districts:")
    chart_feature = st.selectbox("Select Demographic Variable to Plot:", ["Population", "Growth", "Sex-Ratio", "Literacy"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(merged[chart_feature], merged["Total Cog. Crime Under IPC"], alpha=0.7, color="#1f77b4")
    ax.set_title(f"{chart_feature} vs Total Crimes Across Districts", fontsize=12)
    ax.set_xlabel(chart_feature, fontsize=10)
    ax.set_ylabel("Total Cog. Crime Under IPC", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)

with tab3:
    st.header("Data Overview & Filter")
    
    # State selection dropdown
    unique_states = sorted(merged["State"].unique())
    selected_state = st.selectbox("Select State to Filter Data:", unique_states)
    
    # Filter dataset for the selected state
    state_df = merged[merged["State"] == selected_state]
    
    # Calculate summary metrics for the state
    state_population = state_df["Population"].sum()
    state_avg_literacy = state_df["Literacy"].mean()
    state_total_crimes = state_df["Total Cog. Crime Under IPC"].sum()
    
    st.subheader(f"Summary Statistics for {selected_state}")
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric(label="Total Population", value=f"{state_population:,}")
    col_s2.metric(label="Average Literacy Rate", value=f"{state_avg_literacy:.2f}%")
    col_s3.metric(label="Total IPC Crimes", value=f"{int(state_total_crimes):,}")
    
    st.subheader("District Records in Selected State")
    st.dataframe(state_df[["District", "Population", "Growth", "Sex-Ratio", "Literacy", "Total Cog. Crime Under IPC"]])
    
    # Raw previews
    st.subheader("Raw Datasets (Top 5 rows)")
    st.write("Census Dataset Preview:")
    st.dataframe(census.head())
    st.write("Crime Dataset Preview:")
    st.dataframe(crime.head())