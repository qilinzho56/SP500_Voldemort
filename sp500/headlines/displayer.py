# PySimpleGUI referenced https://www.tutorialspoint.com/pysimplegui/index.htm
from sp500.headlines.scraper import headlines
import PySimpleGUI as sg
import webbrowser
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sp500.time_series.visualization.company_profile import company_index_exhibit
from sp500.time_series.visualization.stock_movement import plot_stock_data_interactive
import webbrowser
import os
from pathlib import Path
from tensorflow import keras
import joblib

FONT = {"Arial Bold", 16}
sg.set_options(font=FONT)

DIR = Path(__file__).parents[1] / "time_series"
ANN = keras.models.load_model(DIR / "best_ann_model.h5")
LSTM = keras.models.load_model(DIR / "best_lstm_model.h5")
RNF = joblib.load(DIR /"best_rnf_model.joblib")
# Define a function to draw the matplotlib figure object on the canvas
def draw_figure(canvas, figure):
    tkcanvas = FigureCanvasTkAgg(figure, canvas)
    tkcanvas.draw()
    tkcanvas.get_tk_widget().pack(side='top', fill='both', expand=1)

    return tkcanvas

def save_plotly_figure_as_html(fig, file_path='dynamic_stock_plot.html'):
    fig.write_html(file_path)
    return file_path

def create_row_layout(df, title):
    headings = df.columns.tolist()
    values = df.iloc[0].tolist()
    layout = [[sg.Text(title, justification='center')]]

    for head, value in zip(headings, values):
        layout.append([sg.Text(f"{head}:"), sg.InputText(value, disabled=True)])

    return layout
    


class TickerApplication:
    def __init__(self): 
        """
        Initialized the GUI with windows and parameters that need to be 
        updated continuously.

        window1: user input tickers 
        window2: news and URLs exhibit
        window3: company profile and stock movement plots
        """
        self.tickers = []
        self.news_data = None
        self.window1()
        self.window2 = None
        self.window3 = None

    def window1(self):
        self.lst = sg.Listbox(
            self.tickers, size=(40, 6), expand_y=True, enable_events=True, key="-LIST-"
        )
        layout = [
            [
                sg.Input(size=(40, 6), expand_x=True, key="-INPUT-"),
                sg.Button("Add"),
                sg.Button("Remove"),
                sg.Button("Exit"),
            ],
            [
                sg.Text("Days to Look-up [same for all tickers]"),
                sg.Input(size=(40, 6), key="-Day-"),
            ],
            [self.lst],
            [sg.Text("", key="-MSG-", justification="center")],
            [sg.Submit()],
        ]
        self.window1 = sg.Window("Ticker Application", layout, size=(800, 400))
    
    def run_window1(self, headers):
        while True:
            event, values = self.window1.read()
            if event in (sg.WIN_CLOSED, "Exit"):
                break
            if event == "Add":
                self.tickers.append(values["-INPUT-"])
                self.window1["-LIST-"].update(self.tickers)
                msg = "A new ticker added : {}".format(values["-INPUT-"])
                self.window1["-MSG-"].update(msg)
            if event == "Remove":
                val = self.lst.get()[0]
                self.tickers.remove(val)
                self.window1["-LIST-"].update(self.tickers)
                msg = "A new ticker removed : {}".format(val)
                self.window1["-MSG-"].update(msg)
            if event == "Submit":
                company_list = self.tickers
                self.news_data = headlines(headers, company_list, int(values["-Day-"]))
                if self.news_data.empty:
                    msg = "Ticker Not Found/No Related News!"
                    self.window1["-MSG-"].update(msg)
                else:
                    msg = "Your News Report Has Been Fetched!"
                    self.window1["-MSG-"].update(msg)
                    self.update_window2()
                    self.run_window2()

        self.window1.close()

    def update_window2(self):
        headings = self.news_data.columns.tolist()
        self.data_overview = self.news_data.values.tolist()
        layout = [
            [
                sg.Table(
                    self.data_overview,
                    headings=headings,
                    size=(100, 100),
                    auto_size_columns=False,
                    col_widths=[5, 5, 7, 80, 10],
                    justification="left",
                    expand_x=True,
                    enable_events=True,
                    key="-TABLE-",
                ),
                sg.Button("For More Detailed Analysis"),
            ]
        ]

        self.window2 = sg.Window(
            "Overview with URLs Clickable", layout, size=(1400, 1000)
        )

    def run_window2(self):
        while True:
            event, values = self.window2.read()
            if event == sg.WIN_CLOSED:
                break

            if event == "-TABLE-":
                # First selected row
                row_clicked = values["-TABLE-"][0]
                url = self.data_overview[row_clicked][4]

                if url.startswith("https://"):
                    webbrowser.open(url)

            if event == "For More Detailed Analysis":
                self.update_window3()
                self.run_window3()

        self.window2.close()


    def update_window3(self):
        company_tabs = []
        # a dictionary storing HTML paths for each company
        self.html_paths = {} 

        for company in self.tickers:
            df1, df2, df3, df4, df5 = company_index_exhibit(company)

            company_overview_layout = create_row_layout(df1, f"{company} - Company Overview")
            financial_performance_layout = create_row_layout(df2, f"{company} - Financial Performance")
            cash_flow_layout = create_row_layout(df3, f"{company} - Cash Flow Analysis")
            profitability_efficiency_layout = create_row_layout(df4, f"{company} - Profitability Efficiency Analysis")
            pe_metrics_layout = create_row_layout(df5, f"{company} - PE Metrics")

            img_byte, html_content = plot_stock_data_interactive(company)
            html_file_name = f"{company}_interactive_stock_plot.html"
            html_file_path = Path(__file__).parents[1] / "time_series"/ "visualization" / html_file_name
            
            with open(html_file_path, "w") as html_file:
                html_file.write(html_content)

            self.html_paths[company] = f"file://{html_file_path.resolve()}"

            image_element = sg.Button("Click to see dynamic graph!", image_data=img_byte, 
                          key=f"MOVEIMAGE-{company}",
                          size=(22, 3), 
                          font=("Arial Bold", 15),
                          button_color=("black", "white"))
            
            graph_column = sg.Column([[image_element]], size=(1500, 900))
            
            tab_layout = [[sg.Column(company_overview_layout + financial_performance_layout +
                       cash_flow_layout + profitability_efficiency_layout + 
                       pe_metrics_layout, size=(400, 700)), 
             graph_column],
              [sg.Button("Click for time-series analysis and prediction", 
                            key=f"TIME_SERIES_ANALYSIS-{company}")]
        ]

            company_tab = sg.Tab(company, tab_layout)
            company_tabs.append(company_tab)

        group_layout = [[sg.TabGroup([company_tabs], enable_events=True)]]

        self.window3 = sg.Window("Company Data Overview", group_layout, size=(1800, 1000), finalize=True)


    def run_window3(self):
        while True:
            event, values = self.window3.read()
            if event == sg.WIN_CLOSED:
                break
            
            if isinstance(event, str):
                if event.startswith("MOVEIMAGE-"):
                    company = event.split("MOVEIMAGE-")[1]
                    html_file_path = self.html_paths.get(company)
                    if html_file_path:
                        webbrowser.open(html_file_path)

                if event.startswith("TIME_SERIES_ANALYSIS-"):
                    company = event.split("TIME_SERIES_ANALYSIS-")[1]
                    self.open_window4(company)

        self.window3.close()


    def open_window4(self, company):
        # Example content for Window 4, customize this with actual analysis/prediction content
        layout = [
            [sg.Text(f"Time-series analysis and prediction for {company}")],
            [sg.Button("Close", key="CLOSE_WINDOW4")]
        ]
        window4 = sg.Window(f"Analysis for {company}", layout, modal=True)

        while True:
            event, values = window4.read()
            if event == sg.WIN_CLOSED or event == "CLOSE_WINDOW4":
                break
        window4.close()
