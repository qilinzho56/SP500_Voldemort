from sp500.headlines.displayer import TickerApplication

if __name__ == "__main__":
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "http://finviz.com/quote.ashx?t=",
    }
    display = TickerApplication()
    display.run_window1(headers)
