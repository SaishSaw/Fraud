import streamlit as st
import pickle

print('Successfully executed')

model = pickle.load(open('model.pkl','rb'))

def predict(InsuredAge,
 InsuredZipCode,
 CapitalGains,
 CapitalLoss,
 IncidentTime,
 NumberOfVehicles,
 BodilyInjuries,
 AmountOfInjuryClaim,
 AmountOfPropertyClaim,
 AmountOfVehicleDamage,
 InsurancePolicyNumber,
 CustomerLoyaltyPeriod,
 Policy_Deductible,
 PolicyAnnualPremium,
 UmbrellaLimit):
    prediction = model.predict([[InsuredAge,
 InsuredZipCode,
 CapitalGains,
 CapitalLoss,
 IncidentTime,
 NumberOfVehicles,
 BodilyInjuries,
 AmountOfInjuryClaim,
 AmountOfPropertyClaim,
 AmountOfVehicleDamage,
 InsurancePolicyNumber,
 CustomerLoyaltyPeriod,
 Policy_Deductible,
 PolicyAnnualPremium,
 UmbrellaLimit]])
    if prediction == 0:
        return 'No Fraud '
    else:
        return 'Fraud'
def main():
    st.title("Fraud Insurance Claim")
    html_temp = """
    <div style="background-color:Black;padding:20px">
    <h2 style="color:white;text-align:center;">Streamlit Insurance Fraud Predictor </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    InsuredAge = st.text_input('Insured Age','Type Here')
    InsuredZipCode = st.text_input('Insured Zip Code','Type Here')
    CapitalGains = st.text_input('Capital Gains','Type Here')
    CapitalLoss = st.text_input('Capital Loss','Type Here')
    IncidentTime = st.text_input('Incident Time','Type Here') 
    NumberOfVehicles = st.text_input('Number Of vechiles','Type Here')
    BodilyInjuries = st.text_input('Bodily Injuries','Type Here')
    AmountOfInjuryClaim = st.text_input('Amount of Injury Claim','Type Here')
    AmountOfPropertyClaim = st.text_input('Amount Of Property Claim','Type Here')
    AmountOfVehicleDamage = st.text_input('Amount Of Vehicle Damage','Type Here')
    InsurancePolicyNumber = st.text_input('Insurance Policy Number','Type Here')
    CustomerLoyaltyPeriod = st.text_input('Customer Loyalty Period','Type Here')
    Policy_Deductible = st.text_input('Policy Deductible','Type Here')
    PolicyAnnualPremium = st.text_input('Policy Annual Premium')
    UmbrellaLimit = st.text_input('Umbrella Limit','Type Here')
    result = ""
    if st.button("Predict"):
        result = predict(InsuredZipCode, CapitalGains,CapitalLoss,IncidentTime,NumberOfVehicles,BodilyInjuries,AmountOfInjuryClaim,AmountOfPropertyClaim,AmountOfVehicleDamage,InsurancePolicyNumber,CustomerLoyaltyPeriod,Policy_Deductible,PolicyAnnualPremium,UmbrellaLimit)
        st.success('You Have {}'.format(result))
    if st.button("About"):
        st.text("Lets Detect Fraud")
        st.text("Built with StreamLit")
        st.text("Created By Saish")    
main()
