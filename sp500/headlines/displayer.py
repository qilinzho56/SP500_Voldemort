from sp500.headlines.scraper import headlines
import sys
import PySimpleGUI as sg
import webbrowser

FONT = {"Arial Bold", 16}
sg.set_options(font=FONT)

class TickerApplication:
    def __init__(self):
        self.tickers = []
        self.window1()
        self.window2()

    def window1(self):
        self.lst = sg.Listbox(self.tickers, size=(40, 6),
                 expand_y=True, enable_events=True, key="-LIST-")
        layout = [[sg.Input(size=(40, 6), expand_x=True, key="-INPUT-"), 
                   sg.Button("Add"), sg.Button("Remove"), sg.Button("Exit")],
                [sg.Text("Days to Look-up [same for all tickers]"), sg.Input(size=(40,6), key="-Day-")],
                [self.lst],
                [sg.Text("", key="-MSG-", justification="center")],
                [sg.Submit()]
]
        self.window1 = sg.Window("Ticker Application", layout, size=(800, 400))

    def window2(self):
        self.news_data = None

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
                msg = "Your news report has been fetched!"
                self.window1["-MSG-"].update(msg)

        self.window1.close()

    def update_window2(self):
        headings = self.news_data.columns.tolist()
        self.data_overview = self.news_data.values.tolist()
        layout = [[sg.Table(self.data_overview, headings=headings, size=(100,40), justification="left", 
                    expand_x = True,
                    enable_events=True, key="-TABLE-")]]

        self.window2 = sg.Window("Overview with URLs Clickable", layout, size=(800,400))

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
