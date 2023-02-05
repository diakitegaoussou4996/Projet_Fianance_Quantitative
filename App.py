import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import warnings
import itertools
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import statsmodels.api as sm
import matplotlib
import plotly.express as px
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose

st.set_page_config(
    page_title="Forcast Application for Equities",
    page_icon="💰💸📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

plt.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
# Configuration de la page 

pages_name = ['Presentation', 'Modelisation_ARIMA/Graphiques','Synthèse']
# Create a sidebar with a radio button to select the page
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", pages_name)
st.sidebar.image('https://github.com/aanisoara/Projet_Finance/raw/main/Image/find.gif', width=150, use_column_width ='false')
st.sidebar.markdown("<h1 style='font-family:Lucida Caligraphy;font-size:15px;color:DarkSlateBlue;text-align: center;'> Members: Anisoara ABABII,  Gaoussou DIAKITE,  Eunice KOFFI </h1>", unsafe_allow_html=True)
st.sidebar.markdown("<h1 style='font-family:Lucida Caligraphy;font-size:12px;color:DarkSlateBlue;text-align: center;'> Master 2 Modélistaion Statistiques Economiques et Financières </h1>", unsafe_allow_html=True)
# Importation des données
#@st.cache
def load_data():
    stock = pd.read_excel('https://github.com/aanisoara/Projet_Finance/raw/main/data/Data_projet.xlsx', sheet_name="Returns")
    list1 = pd.read_excel('https://github.com/aanisoara/Projet_Finance/raw/main/data/Data_projet.xlsx', sheet_name="List")
    list1 = list1.drop(list1.index[-1])
    return stock, list1
stock, list1 = load_data()
#--------------------------------------------------------------Sélecteurs dans la partie droite ----------------------------------------------------

if page == 'Presentation':
    st.markdown("<h1 style='font-family:Lucida Caligraphy;font-size:50px;color:Blue;text-align: center;'> Forcast Application for Equities</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='font-family:Lucida Caligraphy;'text-align: center; color:Blue ; font-size:50px;'> Welcome to our financial website !</h1>", unsafe_allow_html=True)
    st.write("""
        <div style="font-family:Lucida Caligraphy;font-size:20px;color:Blue;text-align:center">
        This application website will help you know how your portfolio is performing and of course it gives you a detailed analysis of any stock you might want to invest in! It is composed of three parts, we invite you to explore all of them!        </div>
        """, unsafe_allow_html=True)
    
    list1 = list1[~list1['Stock'].isin(['NTR CT Equity', 'NTR US Equity', 'CTVA US Equity'])]
    add_tile = st.subheader("  Select the filters: ")
    st.write("""
        1. What sector do you want to invest in ?
        """, unsafe_allow_html=True)
    sector_select = st.selectbox(
    '1. What sector do you want to invest in ?',
    (list1.INDUSTRY_SECTOR.unique().tolist()))

    st.write("""
        2. What is the currency you want to invest with ?
        """, unsafe_allow_html=True)
    currency_select = st.selectbox(
    '2. What is the currency you want to invest with ?',
    (list1.CRNCY.unique().tolist()))

    st.write("""
        3. Which country do you want to invest in ?
        """, unsafe_allow_html=True)
    country_select = st.selectbox(
    '3. Which country do you want to invest in ?',
    (list1.CNTRY_OF_DOMICILE.unique().tolist()))

    table = list1[(list1['INDUSTRY_SECTOR'] == sector_select) & (list1['CRNCY'] == currency_select ) & (list1['CNTRY_OF_DOMICILE'] == country_select )]
    st.markdown("<h1 style='font-family:Lucida Caligraphy;font-size:30px;color:DarkSlateBlue;text-align: center;'> Accordind your selections, here are the equities that we dispose: </h1>", unsafe_allow_html=True)
    base = table.Stock.unique().tolist()
    if len(base)==0:
        st.markdown(f"Sorry, we don't find any equities for you. Please change the filters.", unsafe_allow_html=True)
    elif len(base)>=5:
        for elem in base:
            st.markdown(f"<ul><li>{elem}</li></ul>", unsafe_allow_html=True)

    st.write("""<div style="font-family:Lucida Caligraphy;font-size:20px;color:Red;text-align:center">
        Please go to the next page: to view the details of all available actions including those chosen.        
        """, unsafe_allow_html=True)

    ## ________________________________ page 2 ______________________________________

elif page == 'Modelisation_ARIMA/Graphiques':
    #                   ____________________   correlation            ______________

    #st.markdown("<h2 style='text-align: center; color:red ; font-size:20px;'> Interactive Correlation matrix for each action ! </h2>", unsafe_allow_html=True)
    st.write("""<div style="font-family:Lucida Caligraphy;font-size:20px;color:Black;text-align:center">
    Interactive Correlation matrix for each action
    """, unsafe_allow_html=True)
    st.write("""<div style="font-family:Lucida Caligraphy;font-size:20px;color:Red;text-align:center">
        The interactive correlation matrix allows you to visualize the correlation coefficient between each financial asset! Attention ! Be vigilant in selecting the assets in your portfolio, they should not be correlated !
        """, unsafe_allow_html=True)
    
    fig3 = px.imshow(stock.iloc[:, 3:59].corr(),
                labels=dict(x="Columns", y="Columns", color="Correlation"),
                color_continuous_scale=px.colors.diverging.RdBu_r)
    fig3.update_layout(width=800, height=800)
    st.plotly_chart(fig3)

 # _______________

    stock['Dates']=pd.to_datetime(stock.Dates,format='%Y%m%d', errors='ignore')
    cols = ['Dates Future', 'S5ENRS Index - ', 'NTR CT Equity - Basic Materials','NTR US Equity - Basic Materials']
    stock.drop(cols, axis=1, inplace=True)
    stock = stock.sort_values('Dates')
    stock = stock.set_index('Dates')
    stock.index = pd.to_datetime(stock.index)

    liste = stock.columns.to_list()
    stock_selectbox = st.selectbox(
        "Afficher les description d'un actif que vous souhaitez?",
        (liste)
    )

    st.write("""
        <div style="font-family:Lucida Caligraphy;font-size:20px;color:Blue;text-align:center">
        The graphics clearly show that the returns of the selected financial asset are unnstable over the time.
        </div>""", unsafe_allow_html=True)

    col_name = stock_selectbox
    if col_name in stock.columns:
        fig = px.line(x=stock.index, y=stock[col_name], title=col_name)
        fig.update_layout(
            xaxis_title="Index",
            yaxis_title="Value",
            title={
                'text': col_name,
                'font': {'size': 16},
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            }
        )
        st.plotly_chart(fig)

    # ___________________________________________ decomposition de la saisonalité _______________________________

    def plot_seasonal_decompose(result: DecomposeResult, title="Seasonal Decomposition"):
        return (
            make_subplots(
                rows=4,
                cols=1,
                subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"],
            )
            .add_trace(
                go.Scatter(x=result.seasonal.index, y=result.observed, mode="lines"),
                row=1,
                col=1,
            )
            .add_trace(
                go.Scatter(x=result.trend.index, y=result.trend, mode="lines"),
                row=2,
                col=1,
            )
            .add_trace(
                go.Scatter(x=result.seasonal.index, y=result.seasonal, mode="lines"),
                row=3,
                col=1,
            )
            .add_trace(
                go.Scatter(x=result.resid.index, y=result.resid, mode="lines"),
                row=4,
                col=1,
            )
            .update_layout(
                height=900, title=title, margin=dict(t=100), title_x=0.5, showlegend=False
            )
        )

    result = seasonal_decompose(stock[col_name], model='additive')
    st.write("""
    <div style="font-family:Lucida Caligraphy;font-size:20px;color:Blue;text-align:center">
    The graphics clearly show that the returns of financial assets are unstable over the time, as well as their obvious seasonalities. To be precise, the trend provides us with information about the upward and downward movement of the data over a long period; while seasonality stipulates indications of seasonal variance (e.g. decline in asset performance during the Covid period). Then the noises represent peaks and troughs at random intervals.
    </div>""", unsafe_allow_html=True)

    st.write(plot_seasonal_decompose(result, title="Seasonal Decomposition of : "+ col_name))


    #### _______________________________________________ ARIMA Model ____________________________________

    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    l_param = []
    l_param_seasonal=[]
    l_results_aic=[]
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(stock[col_name],
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

                results = mod.fit()
                l_param.append(param)
                l_param_seasonal.append(param_seasonal)
                l_results_aic.append(results.aic)
            except:
                continue

    minimum=l_results_aic[0]
    for i in l_results_aic[1:]:
        if i < minimum: 
            minimum = i
    i=l_results_aic.index(minimum)

    mod = sm.tsa.statespace.SARIMAX(stock[col_name],
                                    order=l_param[i],
                                    seasonal_order=l_param_seasonal[i],
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

    results = mod.fit()

    #results.summary().tables[2]

    st.write("""
    <div style="font-family:Lucida Caligraphy;font-size:20px;color:Blue;text-align:center">
    ARIMA models are denoted by the ARIMA notation (p, d, q). These three parameters take into account seasonality, trend and noise in the data. We generated the possible combinations of these parameters and we selected the minimum AIC according to the statistical criterion.
    An example of standardized residual plots, Histogram, quantile plots, and correlogram is provided in this section.
    </div>""", unsafe_allow_html=True)
    
    plt.rcParams['lines.linewidth'] = 2
    fig = results.plot_diagnostics(figsize=(16, 14))
    st.pyplot(fig)

    #___________________________________  predicted forcast ____________________________________

    pred = results.get_prediction(start=pd.to_datetime('2021-05-31'), dynamic=False)
    pred_ci = pred.conf_int()

    fig2, ax = plt.subplots(figsize=(14, 7))
    stock[col_name]['2017-09-30':].plot(ax=ax, label='observed')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)

    ax.set_xlabel('Dates')
    ax.set_ylabel('Returns')
    plt.rcParams['lines.linewidth'] = 2
    plt.legend()
    plt.title("Validating forecasts")
    st.pyplot(fig2)

    ### ___________________ predicted horozon _________________________


    st.write("""
    <div style="font-family:Lucida Caligraphy;font-size:20px;color:Blue;text-align:center">
    Our model clearly captured the seasonality of close returns. As we plan further into the future, it is natural that we become less confident in our values. 
    This is reflected in the confidence intervals generated by our model, which increase as we move further into the future. 
    Also, the crises and shocks recorded over time (Covid, War in Ukraine) affect our forecasts and the CI.
    </div>""", unsafe_allow_html=True)

    pred_uc = results.get_forecast(steps=70)
    pred_ci = pred_uc.conf_int()
    fig3, ax = plt.subplots(figsize=(14, 7))
    stock[col_name].plot(label='observed')
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.3)
    ax.set_xlabel('Dates')
    ax.set_ylabel('Returns')
    plt.rcParams['lines.linewidth'] = 2
    plt.legend()
    st.pyplot(fig3)

    #___________________________________________ Page 3 _______________________________
else:
    stock['Dates']=pd.to_datetime(stock.Dates,format='%Y%m%d', errors='ignore')
    cols = ['Dates Future', 'S5ENRS Index - ', 'NTR CT Equity - Basic Materials','NTR US Equity - Basic Materials']
    stock.drop(cols, axis=1, inplace=True)
    stock = stock.sort_values('Dates')
    stock = stock.set_index('Dates')
    stock.index = pd.to_datetime(stock.index)

    page3_list = stock.columns.to_list()
    options = st.multiselect(
    'What are your favorite Equities for investing?',
    page3_list)
    if len(options)>=1:
        for element in options:
            st.markdown(f"<ul><li>{element}</li></ul>", unsafe_allow_html=True)
    #st.write('You selected:', options, len(options))

    if len(options)==0:
        st.write("""<div style="font-family:Lucida Caligraphy;font-size:20px;color:Red;text-align:center">
            Please select actions that you prefer !        
            """, unsafe_allow_html=True)
        
    #st.image('Image/frog.gif',width=300, use_column_width ='false')
    st.image('https://github.com/aanisoara/Projet_Finance/raw/main/Image/homme.gif',width=300, use_column_width ='false')
    
    # _____________________ filtre de la table avec ls titres selectionés_______________
    selected_stocks = stock[options]
        # Afficher les données uniquement si l'utilisateur souhaite les voir
    if st.checkbox('Show the dataset'):
        st.subheader('Dataset:')
        st.write(selected_stocks)
    
    #  ____________  poids portofolio equiponderé 
    if len(options)==0:
        st.write("""<div style="font-family:Lucida Caligraphy;font-size:20px;color:Red;text-align:center">
            Please select actions that you prefer !        
            """, unsafe_allow_html=True)
    else:
        w = 1/len(options)
        stock_weight = np.array([w] * len(options))
        ptf_equi = (selected_stocks * stock_weight).sum(axis=1)

        def stat_desc(returns, rf):
            stats = np.cumsum(returns).iloc[-1,]
            stats = np.append(stats, np.std(returns))
            stats = np.append(stats, (np.mean(returns - rf) / np.std(returns)))
            stats = pd.DataFrame(stats, index=["Cumulative Return", "Volatility", "Sharpe Ratio"])
            return stats.T

        results_Q7 = stat_desc(ptf_equi, rf=0.036)
        st.write("""<div style="font-family:Lucida Caligraphy;font-size:20px;color:DarkSlateBlue;text-align:center">
        Cumulative Return;	Volatility	and Sharpe Ratio for our portofolio !     
        """, unsafe_allow_html=True)

        left_column, center_column, right_column = st.columns(3)

        left_column.write("Cumulative Return")
        left_column.write(results_Q7["Cumulative Return"].tolist()[0])

        center_column.write("Volatility")
        center_column.write(results_Q7["Volatility"].tolist()[0])

        right_column.write("Sharpe Ratio")
        right_column.write(results_Q7["Sharpe Ratio"].tolist()[0])

        selected_stock_data = list1[list1['concatenation colonnz stock -industry'].isin(options)]
        #st.write(selected_stock_data)

        data = selected_stock_data.copy()

        ## ________________ graphiques sectors ..

        st.markdown("<hr>", unsafe_allow_html=True)

        st.write("""<div style="font-family:Lucida Caligraphy;font-size:40px;color:DarkSlateBlue;text-align:center">
        Portofolio's caracteristics    
        """, unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        
        
        left_column1, center_column2, right_column3 = st.columns(3)

        with left_column1:
            data = selected_stock_data.copy()
            st.subheader("Sectors")
            plt.xlabel("Sectors")
            plt.ylabel("Count")
            plt.xticks(rotation = 0)
            labels = selected_stock_data['INDUSTRY_SECTOR'].value_counts()
            sns.countplot(x=selected_stock_data['INDUSTRY_SECTOR'])
            st.pyplot(plt)

        with center_column2:
            data = selected_stock_data.copy()
            st.subheader("Currency")
            plt.xlabel("Currency")
            plt.ylabel("Count")
            plt.xticks(rotation = 0)
            labels = selected_stock_data['CRNCY'].value_counts()
            sns.countplot(x=selected_stock_data['CRNCY'])
            st.pyplot(plt)

        with right_column3:
            data = selected_stock_data.copy()
            st.subheader("Country")
            plt.xlabel("Country")
            plt.ylabel("Count")
            plt.xticks(rotation = 0)
            labels = selected_stock_data['CNTRY_OF_DOMICILE'].value_counts()
            sns.countplot(x=selected_stock_data['CNTRY_OF_DOMICILE'])
            st.pyplot(plt)

    st.image('https://github.com/aanisoara/Projet_Finance/raw/main/Image/thanks.gif',width=300, use_column_width ='false')



    
    