import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import ttest_ind



# Carregamento e preparação dos dados
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

# Inicialização do app Dash
app = dash.Dash(__name__)

# Layout do aplicativo
app.layout = html.Div([
    html.Div([
        html.H1("Tesla Stock Analysis", style={'textAlign': 'center'}),
        
        # Dropdown para selecionar a análise
        dcc.Dropdown(
            id='analysis-selector',
            options=[
                {'label': 'Estatísticas Descritivas', 'value': 'descriptive_statistics'},
                {'label': 'Teste T de Student', 'value': 'ttest'},
                {'label': 'Previsão com Regressão Linear', 'value': 'linear_regression'},
                {'label': 'Volume Total de Vendas por Ano', 'value': 'volume_year'},
                {'label': 'Volume Mensal de Vendas', 'value': 'volume_month'},
                {'label': 'Indicadores de Compra e Venda', 'value': 'buy_sell_indicators'},
                {'label': 'Tendência de Preços', 'value': 'price_trend'},
                {'label': 'Correlação Preço x Volume', 'value': 'correlation'}
            ],
            value='descriptive_statistics',  # Valor padrão
            placeholder="Selecione uma análise",
            style={'margin-bottom': '20px'}
        ),
        
        # Gráfico interativo ou texto dinâmico
        dcc.Graph(id='analysis-graph'),
        dcc.Markdown(id='analysis-result', style={'margin-top': '20px', 'font-size': '18px'})
    ], style={'width': '70%', 'margin': '0 auto'}),
    
    html.Hr(),
    
    # Simulação de Investimento
    html.Div([
        html.H2("Simulação de Investimento"),
        # Inputs e botão de simulação de investimento
        html.Div([
            html.Label("Quantidade de Ações:"),
            dcc.Input(id='investment-quantity', type='number', value=100, style={'margin-bottom': '10px'}),
            
            html.Label("Data de Compra:"),
            dcc.DatePickerSingle(
                id='investment-date',
                min_date_allowed=df['Date'].min(),
                max_date_allowed=df['Date'].max(),
                initial_visible_month=df['Date'].min(),
                date=df['Date'].min()
            ),
            
            html.Button('Simular Investimento', id='simulate-button', n_clicks=0, style={'margin-top': '10px'})
        ], style={'margin-bottom': '20px'}),
        
        dcc.Graph(id='investment-graph'),
        dcc.Markdown(id='investment-result', style={'margin-top': '20px', 'font-size': '18px'})
    ], style={'width': '70%', 'margin': '0 auto'})
])


@app.callback(
    [Output('analysis-graph', 'figure'),
     Output('analysis-result', 'children')],
    [Input('analysis-selector', 'value')]
)
def update_analysis(selected_analysis):
    # Estatísticas Descritivas
    if selected_analysis == 'descriptive_statistics':
        # Regra de Sturges
        num_classes = int(1 + 3.322 * np.log10(len(df)))
        fig = px.histogram(df, x='Close', nbins=num_classes, title='Histograma de Preços (Fechamento)')
        
        # Texto de resumo
        result_text = (
            f"- **Média**: {df['Close'].mean():.2f}\n"
            f"- **Mediana**: {df['Close'].median():.2f}\n"
            f"- **Moda**: {df['Close'].mode()[0]:.2f}\n"
            f"- **Variância**: {df['Close'].var():.2f}\n"
            f"- **Desvio Padrão**: {df['Close'].std():.2f}\n"
            f"- **Desvio Médio Absoluto**: {np.mean(np.abs(df['Close'] - df['Close'].mean())):.2f}"
        )
    
    # Teste T de Student
    elif selected_analysis == 'ttest':
        df_2015 = df[df['Year'] == 2015]['Close']
        df_2020 = df[df['Year'] == 2020]['Close']
        t_stat, p_value = ttest_ind(df_2015, df_2020, equal_var=False)
        
        # Sem gráfico para teste T
        fig = go.Figure()
        result_text = (
            f"- **Estatística t**: {t_stat:.2f}\n"
            f"- **Valor p**: {p_value:.2f}\n"
            f"{'As médias são significativamente diferentes.' if p_value < 0.05 else 'Não há diferença significativa entre as médias.'}"
        )
    
    # Previsão com Regressão Linear
    elif selected_analysis == 'linear_regression':
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
        
        result_text = f"- **Erro Quadrático Médio (MSE)**: {mse:.2f}"
    
    # Volume Total de Vendas por Ano
    elif selected_analysis == 'volume_year':
        sales_by_year = df.groupby('Year')['Volume'].sum().reset_index()
        fig = px.bar(sales_by_year, x='Year', y='Volume', title='Volume Total de Vendas por Ano')
        result_text = ""
    
    # Volume Mensal de Vendas
    elif selected_analysis == 'volume_month':
        sales_by_month = df.groupby(['Year', 'Month'])['Volume'].sum().reset_index()
        fig = px.line(sales_by_month, x='Month', y='Volume', color='Year', title='Volume Mensal de Vendas')
        result_text = ""
    
    # Indicadores de Compra e Venda
    elif selected_analysis == 'buy_sell_indicators':
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Preço de Fechamento'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], mode='lines', name='SMA 20 Dias'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], mode='lines', name='SMA 50 Dias'))
        fig.update_layout(title='Indicadores de Compra e Venda', xaxis_title='Data', yaxis_title='Preço')
        result_text = ""
    
    # Tendência de Preços
    elif selected_analysis == 'price_trend':
        fig = px.line(df, x='Date', y='Close', title='Tendência de Preços')
        result_text = ""
    
    # Correlação Preço x Volume
    elif selected_analysis == 'correlation':
        fig = px.scatter(df, x='Volume', y='Close', title='Correlação Preço x Volume')
        result_text = ""
    
    else:
        # Caso nenhum valor seja selecionado
        fig = go.Figure()
        result_text = "🚨 **Erro**: Selecione uma análise válida."
    
    return fig, result_text

@app.callback(
    [Output('investment-graph', 'figure'),
     Output('investment-result', 'children')],
    [Input('simulate-button', 'n_clicks')],
    [State('investment-quantity', 'value'),
     State('investment-date', 'date')]
)
def simulate_investment(n_clicks, quantity, purchase_date):
    # Não atualiza até o botão ser clicado
    if n_clicks == 0:
        raise PreventUpdate
    
    # Validação dos inputs
    if not quantity or quantity <= 0:
        return go.Figure(), "🚨 **Erro**: Insira uma quantidade válida de ações (maior que 0)."
    
    if not purchase_date:
        return go.Figure(), "🚨 **Erro**: Selecione uma data válida de compra."
    
    # Converter data de compra
    purchase_date = pd.to_datetime(purchase_date)
    
    # Verificar se a data está no dataset
    if purchase_date not in df['Date'].values:
        return go.Figure(), "🚨 **Erro**: Data de compra inválida ou fora do intervalo de dados."
    
    # Filtrar o preço de fechamento na data de compra
    purchase_price = df.loc[df['Date'] == purchase_date, 'Close'].values[0]
    
    # Cálculo do custo inicial
    initial_cost = quantity * purchase_price
    
    # Valor atual do investimento com base no preço mais recente
    latest_price = df['Close'].iloc[-1]
    current_value = quantity * latest_price
    
    # Diferença entre o valor atual e o custo inicial
    profit_loss = current_value - initial_cost
    
    # Texto de resultado formatado em bullets
    result_text = (
        f"📅 **Data de Compra**: {purchase_date.date()}\n"
        f"- 💵 **Preço na Época**: ${purchase_price:,.2f}\n"
        f"- 📊 **Quantidade de Ações Compradas**: {quantity}\n"
        f"- 🔢 **Custo Inicial**: ${initial_cost:,.2f}\n"
        f"- 💰 **Preço Atual das Ações**: ${latest_price:,.2f}\n"
        f"- 📈 **Valor Atual das Ações**: ${current_value:,.2f}\n"
        f"- 📉 **Lucro/Prejuízo**: ${profit_loss:,.2f} {'(Lucro)' if profit_loss > 0 else '(Prejuízo)'}"
    )
    
    # Simulação do valor do investimento ao longo do tempo
    df['Investment Value'] = quantity * df['Cumulative Returns']
    
    # Gráfico de evolução do investimento
    fig = px.line(df, x='Date', y='Investment Value', title='Evolução do Investimento')
    fig.update_layout(xaxis_title='Data', yaxis_title='Valor do Investimento')
    
    return fig, result_text

if __name__ == '__main__':
    app.run_server(debug=True)
