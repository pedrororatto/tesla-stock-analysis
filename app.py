import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import ttest_ind

st.set_page_config(page_title="Tesla Stock Analysis", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('Dataset/TSLA.csv')
    df = df.dropna()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Signal'] = np.where(df['SMA_20'] > df['SMA_50'], 'Comprar', 'Vender')
    df['Returns'] = df['Close'].pct_change()
    df['Cumulative Returns'] = (1 + df['Returns']).cumprod()
    return df

df = load_data()

st.title("Tesla Stock Analysis 📈")

analysis_options = [
    "Estatísticas Descritivas",
    "Teste T de Student",
    "Previsão com Regressão Linear",
    "Volume Total de Vendas por Ano",
    "Volume Mensal de Vendas",
    "Indicadores de Compra e Venda",
    "Tendência de Preços",
    "Correlação Preço x Volume",
    "Simulação de Investimento"
]

st.sidebar.title("Sobre o Projeto")
st.sidebar.markdown("Este é um projeto de análise de dados de ações da Tesla, desenvolvido com Streamlit, para a disciplina de Ciência de Dados")
st.sidebar.markdown("Autor: [Pedro Roratto]( https://github.com/pedrororatto )")
selected_analysis = st.sidebar.selectbox("Selecione a análise desejada:", analysis_options)

st.sidebar.subheader("Sobre o Dataset")
st.sidebar.markdown("O dataset utilizado contém informações sobre o preço de fechamento das ações da Tesla, bem como o volume de vendas, entre outros indicadores.")
st.sidebar.markdown(f"**Período de Análise**: {df['Date'].dt.date.min()} a {df['Date'].dt.date.max()}")
st.sidebar.markdown(f"**Total de Dias**: {len(df)}")

st.sidebar.subheader("Legenda")
st.sidebar.markdown(" - **Open**: Preço de abertura da ação")
st.sidebar.markdown(" - **High**: Maior preço da ação no dia")
st.sidebar.markdown(" - **Low**: Menor preço da ação no dia")
st.sidebar.markdown(" - **Close**: Preço de fechamento da ação")
st.sidebar.markdown(" - **Volume**: Volume de ações negociadas")
st.sidebar.markdown(" - **Adj Close**: Preço de fechamento ajustado")

st.sidebar.write(df)

def descriptive_statistics():
    st.subheader("Estatísticas Descritivas")
    
    num_classes = int(1 + 3.322 * np.log10(len(df)))
    fig = px.histogram(df, x='Close', nbins=num_classes, title='Histograma de Preços (Fechamento)')
    st.plotly_chart(fig, use_container_width=True)
    st.write("O histograma sugere que a maioria dos preços de fechamento estava em um intervalo baixo (próximo a 0-100), mas em certos momentos, o preço atingiu valores muito mais altos. Essa variação pode refletir flutuações no desempenho da Tesla ou condições de mercado. O formato do gráfico é típico de distribuições com valores extremos, que deslocam a média para cima, mas não afetam tanto a moda ou a mediana.")


    mad = np.mean(np.abs(df['Close'] - df['Close'].mean()))
    stats_text = f"""
    - **Média**: {df['Close'].mean():.2f} - Representa o preço médio das ações ao longo do período analisado. Isso indica que, em média, o preço das ações foi de aproximadamente $105.92.  
    - **Mediana**: {df['Close'].median():.2f} - Mostra o valor central dos dados quando ordenados. Metade dos preços foi menor que $45.92, e a outra metade foi maior.
    - **Moda**: {df['Close'].mode()[0]:.2f} - É o valor mais frequente nos dados. Isso pode indicar períodos em que o preço permaneceu estável em torno de $5.48.
    - **Variância**: {df['Close'].var():.2f} - Mede a dispersão dos dados em relação à média. Valores altos indicam grande variabilidade nos preços das ações.
    - **Desvio Padrão**: {df['Close'].std():.2f} - É a raiz quadrada da variância. Indica o quanto os preços variam em relação à média.
    - **Desvio Médio Absoluto**: {mad:.2f} - É a média das distâncias entre os preços e a média. É uma medida de dispersão mais robusta que o desvio padrão.
    """
    st.markdown(stats_text)

def ttest_analysis():
    st.subheader("Teste T de Student")
    df_2015 = df[df['Year'] == 2015]['Close']
    df_2020 = df[df['Year'] == 2020]['Close']
    t_stat, p_value = ttest_ind(df_2015, df_2020, equal_var=False)

    st.write("Foram usados os preços de fechamento das ações da Tesla dos anos de 2015 e 2020.")
    st.markdown(f"""
    - **Estatística t**: {t_stat:.2f}  
    - **Valor p**: {p_value:.4f}  
    {'As médias são significativamente diferentes.' if p_value < 0.05 else 'Não há diferença significativa entre as médias.'}
    """)

    st.markdown('**Esses resultados mostram que os preços médios das ações em 2015 e 2020 eram diferentes, o que pode indicar mudanças substanciais no mercado ou no desempenho da Tesla nesses períodos.**')

def linear_regression_analysis():
    st.subheader("Previsão com Regressão Linear")
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    X = df[['Days']]
    y = df['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_test['Days'], y=y_test, mode='markers', name='Dados Reais'))
    fig.add_trace(go.Scatter(x=X_test['Days'], y=y_pred, mode='lines', name='Previsão'))
    fig.update_layout(title='Previsão com Regressão Linear', xaxis_title='Dias', yaxis_title='Preço de Fechamento')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"- **Erro Quadrático Médio (MSE)**: {mse:.2f}")

    st.subheader('Conclusão')
    st.markdown(f"O modelo conseguiu capturar a tendência geral dos preços ao longo do tempo, mas há algum nível de erro devido à variabilidade natural do mercado. O MSE de {mse:.2f} sugere que, em média, há uma grande diferença entre os preços previstos e os reais.")

def investment_simulation():
    st.subheader("Simulação de Investimento")
    quantity = st.number_input("Quantidade de ações compradas:", min_value=1, value=100, step=1)
    purchase_date = st.date_input("Data de compra:", min_value=df['Date'].min().date(), max_value=df['Date'].max().date(), value=df['Date'].min().date())

    if st.button("Simular"):
        purchase_date = pd.to_datetime(purchase_date)
        if purchase_date not in df['Date'].values:
            st.error("🚨 Data inválida ou fora do intervalo de dados.")
            return

        purchase_price = df.loc[df['Date'] == purchase_date, 'Close'].values[0]
        initial_cost = quantity * purchase_price
        latest_price = df['Close'].iloc[-1]
        current_value = quantity * latest_price
        profit_loss = current_value - initial_cost

        result_text = f"""
        - 📅 **Data de Compra**: {purchase_date.date()}  
        - 💵 **Preço na Época**: ${purchase_price:,.2f}  
        - 📊 **Quantidade de Ações Compradas**: {quantity}  
        - 🔢 **Custo Inicial**: ${initial_cost:,.2f}  
        - 💰 **Preço Atual das Ações**: ${latest_price:,.2f}  
        - 📈 **Valor Atual das Ações**: ${current_value:,.2f}  
        - 📉 **Lucro/Prejuízo**: ${profit_loss:,.2f} {'(Lucro)' if profit_loss > 0 else '(Prejuízo)'} 
        - 📈 **Rentabilidade**: {profit_loss / initial_cost:.2%}
        """
        st.markdown(result_text)

        purchase_date = pd.to_datetime(purchase_date)
        filtered_df = df[df['Date'] >= purchase_date]

        filtered_df['Investment Value'] = quantity * filtered_df['Cumulative Returns']

        fig = px.line(
            filtered_df, 
            x='Date', 
            y='Investment Value', 
            title=f"Evolução do Investimento a partir de {purchase_date.date()}"
        )

        st.plotly_chart(fig, use_container_width=True)

if selected_analysis == "Estatísticas Descritivas":
    descriptive_statistics()
elif selected_analysis == "Teste T de Student":
    ttest_analysis()
elif selected_analysis == "Previsão com Regressão Linear":
    linear_regression_analysis()
elif selected_analysis == "Simulação de Investimento":
    investment_simulation()
elif selected_analysis == "Volume Total de Vendas por Ano":
    st.subheader("Volume Total de Vendas por Ano")
    st.write("O volume total de vendas é a soma de todas as ações negociadas em um determinado ano.")
    sales_by_year = df.groupby('Year')['Volume'].sum().reset_index()
    fig = px.bar(sales_by_year, x='Year', y='Volume', title='Volume Total de Vendas por Ano')
    st.plotly_chart(fig, use_container_width=True)
elif selected_analysis == "Volume Mensal de Vendas":
    st.subheader("Volume Mensal de Vendas")
    st.write("O volume mensal de vendas é a soma de todas as ações negociadas em um determinado mês.")
    sales_by_month = df.groupby(['Year', 'Month'])['Volume'].sum().reset_index()
    fig = px.line(sales_by_month, x='Month', y='Volume', color='Year', title='Volume Mensal de Vendas')
    st.plotly_chart(fig, use_container_width=True)
elif selected_analysis == "Indicadores de Compra e Venda":
    st.subheader("Indicadores de Compra e Venda")
    st.write("Os indicadores de compra e venda são baseados em médias móveis (SMA). Quando a SMA de 20 dias cruza acima da de 50 dias, isso é um sinal de compra. Quando cruza abaixo, é um sinal de venda.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Preço de Fechamento'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], mode='lines', name='SMA 20 Dias'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], mode='lines', name='SMA 50 Dias'))
    fig.update_layout(title='Indicadores de Compra e Venda', xaxis_title='Data', yaxis_title='Preço')
    st.plotly_chart(fig, use_container_width=True)
elif selected_analysis == "Tendência de Preços":
    st.subheader("Tendência de Preços")
    st.write("Aqui, podemos ver a evolução dos preços de fechamento das ações da Tesla ao longo do tempo.")
    fig = px.line(df, x='Date', y='Close', title='Tendência de Preços')
    st.plotly_chart(fig, use_container_width=True)
elif selected_analysis == "Correlação Preço x Volume":
    st.subheader("Correlação Preço x Volume")
    st.write("Podemos ver a relação entre o preço de fechamento e o volume de vendas das ações da Tesla.")
    fig = px.scatter(df, x='Volume', y='Close', title='Correlação Preço x Volume')
    st.plotly_chart(fig, use_container_width=True)
