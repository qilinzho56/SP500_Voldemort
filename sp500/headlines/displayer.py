# PySimpleGUI referenced https://www.tutorialspoint.com/pysimplegui/index.htm
from sp500.headlines.scraper import headlines
import PySimpleGUI as sg
import webbrowser
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sp500.time_series.time_series import load_ticker_data, time_rnf_model
from sp500.visualization.company_profile import company_index_exhibit

FONT = {"Arial Bold", 16}
sg.set_options(font=FONT)

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
                else:
                    msg = "Your News Report Has Been Fetched!"
                self.window1["-MSG-"].update(msg)

        self.window1.close()

    def window2(self):
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

        self.window2.close()

    def create_row_layout(self, df, title):
        headings = df.columns.tolist()
        values = df.iloc[0].tolist()
        layout = [[sg.Text(title, justification='center')]]
        for head, value in zip(headings, values):
            layout.append([sg.Text(f"{head}:"), sg.InputText(value, disabled=True)])
        return layout

    def window3(self):
        company_tabs = []

        for company in self.tickers:
            df1, df2, df3, df4, df5 = company_index_exhibit(company)

            company_overview_layout = self.create_row_layout(df1, f"{company} - Company Overview")
            financial_performance_layout = self.create_row_layout(df2, f"{company} - Financial Performance")
            cash_flow_layout = self.create_row_layout(df3, f"{company} - Cash Flow Analysis")
            profitability_efficiency_layout = self.create_row_layout(df4, f"{company} - Profitability Efficiency Analysis")
            pe_metrics_layout = self.create_row_layout(df5, f"{company} - PE Metrics")

            tab_layout = company_overview_layout + financial_performance_layout + cash_flow_layout + profitability_efficiency_layout + pe_metrics_layout

            company_tab = sg.Tab(company, tab_layout)
            company_tabs.append(company_tab)

        group_layout = [[sg.TabGroup([company_tabs], enable_events=True)]]

        self.window3 = sg.Window("Company Data Overview", group_layout, size=(1400, 1000), finalize=True)

    def run_window3(self):
        while True:
            event, values = self.window3.read()
            if event == sg.WIN_CLOSED:
                break
        self.window3.close()

