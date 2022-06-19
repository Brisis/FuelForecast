import math
from tkinter import *
import tkinter.messagebox
import customtkinter
import pandas as pd
import get_forecast

# from petrol import getFuture

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

# root = customtkinter.CTk()
# root.title('Fuel Forecasting by Simon Mhere')
# root.geometry("1200x800")
# load dataset
df = pd.read_csv("assets/dataset.csv")


class App(customtkinter.CTk):
    WIDTH = 1200
    HEIGHT = 800

    def __init__(self):
        super().__init__()

        self.title("Fuel Forecasting")
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)  # call .on_closing() when app gets closed

        # ============ create two frames ============

        # configure grid layout (2x1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.frame_left = customtkinter.CTkFrame(master=self,
                                                 width=180,
                                                 corner_radius=0)
        self.frame_left.grid(row=0, column=0, sticky="nswe")

        self.frame_right = customtkinter.CTkFrame(master=self)
        self.frame_right.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)

        # ============ frame_left ============

        # configure grid layout (1x11)
        self.frame_left.grid_rowconfigure(0, minsize=10)  # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(5, weight=1)  # empty row as spacing
        self.frame_left.grid_rowconfigure(8, minsize=20)  # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(11, minsize=10)  # empty row with minsize as spacing

        self.label_1 = customtkinter.CTkLabel(master=self.frame_left,
                                              text="Fuel Forecasting",
                                              text_font=("Roboto Medium", -16))  # font name and size in px
        self.label_1.grid(row=1, column=0, pady=10, padx=10)

        self.button_1 = customtkinter.CTkButton(master=self.frame_left,
                                                text="Petrol Forecast",
                                                fg_color=("gray75", "gray30"),  # <- custom tuple-color
                                                command=self.getPetrolPrediction)
        self.button_1.grid(row=2, column=0, pady=10, padx=20)

        self.button_2 = customtkinter.CTkButton(master=self.frame_left,
                                                text="Diesel Forecast",
                                                fg_color=("gray75", "gray30"),  # <- custom tuple-color
                                                command=self.getDieselPrediction)
        self.button_2.grid(row=3, column=0, pady=10, padx=20)

        # self.button_3 = customtkinter.CTkButton(master=self.frame_left,
        #                                         text="Download",
        #                                         fg_color=("gray75", "gray30"),  # <- custom tuple-color
        #                                         command=self.button_event)
        # self.button_3.grid(row=4, column=0, pady=10, padx=20)

        self.switch_2 = customtkinter.CTkSwitch(master=self.frame_left,
                                                text="Dark Mode",
                                                command=self.change_mode)
        self.switch_2.grid(row=10, column=0, pady=10, padx=20, sticky="w")

        # ============ frame_right ============

        # configure grid layout (3x7)
        self.frame_right.rowconfigure((0, 1, 2, 3), weight=1)
        self.frame_right.rowconfigure(7, weight=10)
        self.frame_right.columnconfigure((0, 1), weight=1)
        self.frame_right.columnconfigure(2, weight=0)

        self.frame_info = customtkinter.CTkFrame(master=self.frame_right)
        self.frame_info.grid(row=0, column=0, columnspan=2, rowspan=4, pady=20, padx=20, sticky="nsew")

        # ============ frame_info ============

        # configure grid layout (1x1)
        self.frame_info.rowconfigure(0, weight=2)
        self.frame_info.columnconfigure(0, weight=1)

        self.label_info_1 = customtkinter.CTkLabel(master=self.frame_info,
                                                   text="Data Reader",
                                                   height=50,
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.LEFT)
        self.label_info_1.grid(column=0, row=0, sticky="nwe", padx=15, pady=10)

        self.csv_viewer = Text(self.frame_info, height=28, width=65, wrap=WORD, fg="silver", font=13, pady=10)
        self.csv_viewer.grid(column=0, row=1, sticky="nwe", padx=15, pady=10)

        # self.csv_viewer = customtkinter.CTkFrame(master=self.frame_info,
        #                                          text=df,
        #                                          height=400,
        #                                          fg_color=("white", "gray38"),  # <- custom tuple-color
        #                                          justify=tkinter.LEFT)
        #
        # self.csv_viewer.grid(column=0, row=1, sticky="nwe", padx=15, pady=15)

        # ============ frame_right ============

        self.radio_var = tkinter.IntVar(value=0)

        self.label_radio_group = customtkinter.CTkLabel(master=self.frame_right,
                                                        text="Get Forecast Report")
        self.label_radio_group.grid(row=0, column=2, columnspan=1, pady=20, padx=10, sticky="")

        self.checkbox_button_1 = customtkinter.CTkButton(master=self.frame_right,
                                                         height=25,
                                                         text="Download Report",
                                                         border_width=3,  # <- custom border_width
                                                         fg_color=None,  # <- no fg_color
                                                         command=self.getReport)
        self.checkbox_button_1.grid(row=1, column=2, pady=10, padx=20, sticky="n")

        self.entry = customtkinter.CTkEntry(master=self.frame_right,
                                            width=120,
                                            placeholder_text="Enter number of months")
        self.entry.grid(row=8, column=0, columnspan=2, pady=20, padx=20, sticky="we")

        self.button_5 = customtkinter.CTkButton(master=self.frame_right,
                                                text="Forecast",
                                                command=self.getPrediction)
        self.button_5.grid(row=8, column=2, columnspan=1, pady=20, padx=20, sticky="we")
        self.display_data()

    def change_mode(self):
        if self.switch_2.get() == 1:
            customtkinter.set_appearance_mode("dark")
        else:
            customtkinter.set_appearance_mode("light")

    def on_closing(self, event=0):
        self.destroy()

    def display_data(self):
        # Clear The text
        self.csv_viewer.delete(1.0, END)
        self.csv_viewer.insert(END,
                               '\t' + 'Date' + '\t\t\t' + 'Petrol Quantity' + '\t\t\t' + 'Diesel Quantity' + '\n\n')

        date_values = []
        for i in df['Month']:
            date_values.append(i)

        petrol_values = []
        for j in df['Petrol']:
            petrol_values.append(j)

        diesel_values = []
        for k in df['Diesel']:
            diesel_values.append(k)

        for i in range(len(date_values)):
            self.csv_viewer.insert(END, '\t' + str(date_values[i]) + '\t\t\t' + str(petrol_values[i]) + '\t\t\t' + str(
                diesel_values[i]) + '\n\n')

    results = ""
    isPetrolMode = True

    def getDieselPrediction(self):
        print("Choose Diesel Prediction")
        # Clear The text
        self.csv_viewer.delete(1.0, END)
        self.csv_viewer.insert(END, '\t' + 'Date' + '\t\t\t' + 'Diesel Quantity' + '\n\n')

        self.isPetrolMode = False

        date_values = []
        for i in df['Month']:
            date_values.append(i)

        diesel_values = []
        for k in df['Diesel']:
            diesel_values.append(k)

        for i in range(len(date_values)):
            self.csv_viewer.insert(END, '\t' + str(date_values[i]) + '\t\t\t' + str(
                diesel_values[i]) + '\n\n')

        # get_forecast.getForecast(6, "Diesel")

    def getPetrolPrediction(self):
        print("Choose Petrol Prediction")
        # Clear The text
        self.csv_viewer.delete(1.0, END)
        self.csv_viewer.insert(END, '\t' + 'Date' + '\t\t\t' + 'Petrol Quantity' + '\n\n')

        isPetrolMode = True

        date_values = []
        for i in df['Month']:
            date_values.append(i)

        petrol_values = []
        for j in df['Petrol']:
            petrol_values.append(j)

        for i in range(len(date_values)):
            self.csv_viewer.insert(END, '\t' + str(date_values[i]) + '\t\t\t' + str(
                petrol_values[i]) + '\t\t\t' + '\n\n')

    def getPrediction(self):
        print("Now Forecasting")
        # Clear The text
        self.csv_viewer.delete(1.0, END)
        search_value = self.entry.get()

        if self.isPetrolMode:
            get_forecast.getForecast(int(search_value), "Petrol")

            forecasted_results = pd.read_csv("assets/forecasted/petrol_forecasted_data.csv")
            # forecasted_results.rename(columns={"": "Month", "Forecast": "Petrol Quantity Forecast"})
            forecasted_results.set_axis(["Month", "Petrol Quantity Forecast"], axis=1, inplace=True)

            self.results = forecasted_results

            # Clear The text
            self.csv_viewer.delete(1.0, END)
            self.csv_viewer.insert(END, '\t' + 'Date' + '\t\t\t' + 'Petrol Quantity Forecast' + '\n\n')

            date_values = []
            for i in forecasted_results['Month']:
                date_values.append(i)

            petrol_values = []
            for j in forecasted_results['Petrol Quantity Forecast']:
                formatted = math.ceil(j)
                petrol_values.append(formatted)

            for i in range(len(date_values)):
                self.csv_viewer.insert(END, '\t' + str(date_values[i]) + '\t\t\t' + str(
                    petrol_values[i]) + '\t\t\t' + '\n\n')

        else:
            get_forecast.getForecast(int(search_value), "Diesel")

            forecasted_results = pd.read_csv("assets/forecasted/diesel_forecasted_data.csv")
            # forecasted_results.rename(columns={"": "Month", "Forecast": "Petrol Quantity Forecast"})
            forecasted_results.set_axis(["Month", "Diesel Quantity Forecast"], axis=1, inplace=True)

            self.results = forecasted_results

            # Clear The text
            self.csv_viewer.delete(1.0, END)
            self.csv_viewer.insert(END, '\t' + 'Date' + '\t\t\t' + 'Diesel Quantity Forecast' + '\n\n')

            date_values = []
            for i in forecasted_results['Month']:
                date_values.append(i)

            diesel_values = []
            for j in forecasted_results['Diesel Quantity Forecast']:
                formatted = math.ceil(j)
                diesel_values.append(formatted)

            for i in range(len(date_values)):
                self.csv_viewer.insert(END, '\t' + str(date_values[i]) + '\t\t\t' + str(
                    diesel_values[i]) + '\t\t\t' + '\n\n')

    def getReport(self):
        print("Downloaded Report")
        self.results.to_csv('assets/forecasted/reports/forecast_report.csv')


if __name__ == "__main__":
    app = App()
    app.mainloop()
