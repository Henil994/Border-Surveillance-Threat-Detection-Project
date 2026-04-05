import tkinter as tk
from tkinter import ttk
import random
import time
import math
import folium
import subprocess
import winsound


class SurveillanceDashboard:

    def __init__(self, root):

        self.root = root
        self.root.title("AI BORDER DEFENCE COMMAND CENTER - ELITE")
        self.root.geometry("1600x900")
        self.root.configure(bg="#001a2c")

        self.running = False
        self.angle = 0

        self.total = 0
        self.human = 0
        self.drone = 0
        self.animal = 0
        self.noise = 0

        self.targets = []

        # ------------------------
        # BORDER LOCATIONS
        # ------------------------

        self.locations = {
            "Kashmir Sector":[34.0837,74.7973],
            "Punjab Border":[31.1471,75.3412],
            "Rajasthan Border":[26.9124,75.7873],
            "Gujarat Border":[23.0225,72.5714],
            "Ladakh Sector":[34.1526,77.5771],
            "Arunachal Border":[28.2180,94.7278],
            "Nagaland Border":[26.1584,94.5624],
            "Manipur Border":[24.6637,93.9063],
            "Mizoram Border":[23.1645,92.9376],
            "Tripura Border":[23.9408,91.9882],
            "Sikkim Border":[27.5330,88.5122],
            "Uttarakhand Border":[30.0668,79.0193],
            "Himachal Border":[31.1048,77.1734],
            "Kutch Sector":[23.7337,69.8597],
            "Barmer Sector":[25.7500,71.3833],
            "Jaisalmer Sector":[26.9157,70.9083],
            "Tawang Sector":[27.5860,91.8590],
            "Uri Sector":[34.0806,74.0500],
            "Poonch Sector":[33.7706,74.0900],
            "Kargil Sector":[34.5553,76.1340]
        }

        # ------------------------
        # RADAR POSITION MAP
        # ------------------------

        self.radar_positions = {

            "Kashmir Sector": (100,40),
            "Ladakh Sector": (120,30),
            "Himachal Border": (90,50),

            "Arunachal Border": (170,50),
            "Nagaland Border": (160,80),
            "Manipur Border": (170,100),

            "Sikkim Border": (150,70),
            "Tripura Border": (170,130),
            "Mizoram Border": (160,150),

            "Punjab Border": (40,80),
            "Rajasthan Border": (30,120),
            "Gujarat Border": (50,150),

            "Kutch Sector": (60,160),
            "Barmer Sector": (50,140),
            "Jaisalmer Sector": (45,130),

            "Uri Sector": (70,40),
            "Poonch Sector": (80,50),
            "Kargil Sector": (110,35),

            "Tawang Sector": (170,60),
            "Uttarakhand Border": (120,60)
        }

        self.build_ui()
        self.animate_radar()

    # ---------------------------------------------------
    # UI
    # ---------------------------------------------------

    def build_ui(self):

        header = tk.Frame(self.root,bg="#00b3b3",height=70)
        header.pack(fill="x")

        tk.Label(
            header,
            text="📡 AI BORDER SURVEILLANCE COMMAND CENTER",
            font=("Consolas",20,"bold"),
            bg="#00b3b3"
        ).pack(pady=10)

        controls = tk.Frame(self.root,bg="#001a2c")
        controls.pack(pady=10)

        tk.Button(
            controls,text="START MONITORING",
            bg="#00ff9c",font=("Consolas",12,"bold"),
            width=18,command=self.start_system
        ).grid(row=0,column=0,padx=20)

        tk.Button(
            controls,text="STOP SYSTEM",
            bg="#ff3b3b",font=("Consolas",12,"bold"),
            width=18,command=self.stop_system
        ).grid(row=0,column=1)

        self.status=tk.Label(
            self.root,text="STATUS: STOPPED",
            fg="#00ff9c",bg="#001a2c",
            font=("Consolas",18,"bold")
        )
        self.status.pack()

        main=tk.Frame(self.root,bg="#001a2c")
        main.pack(fill="both",expand=True)
        main.columnconfigure(1,weight=1)

        # LEFT PANEL

        left=tk.Frame(main,bg="#00263e",width=260)
        left.grid(row=0,column=0,sticky="ns")

        tk.Label(left,text="LIVE DEFENCE ANALYTICS",
                 fg="#00e6e6",bg="#00263e",
                 font=("Consolas",13,"bold")).pack(pady=10)

        self.total_label=tk.Label(left,text="Total Predictions: 0",bg="#00263e",fg="white")
        self.total_label.pack(anchor="w",padx=10)

        self.human_label=tk.Label(left,text="Human Intrusion: 0",fg="red",bg="#00263e")
        self.human_label.pack(anchor="w",padx=10)

        self.drone_label=tk.Label(left,text="Drone Threat: 0",fg="yellow",bg="#00263e")
        self.drone_label.pack(anchor="w",padx=10)

        self.animal_label=tk.Label(left,text="Animal Threat: 0",fg="#00ffcc",bg="#00263e")
        self.animal_label.pack(anchor="w",padx=10)

        self.noise_label=tk.Label(left,text="Environmental Noise: 0",fg="#66ccff",bg="#00263e")
        self.noise_label.pack(anchor="w",padx=10)

        tk.Label(left,text="AI MODEL INTELLIGENCE",
                 fg="#00e6e6",bg="#00263e",
                 font=("Consolas",13,"bold")).pack(pady=15)

        tk.Label(left,text="Human Probability",fg="red",bg="#00263e").pack()
        self.human_bar=ttk.Progressbar(left,length=180)
        self.human_bar.pack(pady=3)

        tk.Label(left,text="Drone Probability",fg="yellow",bg="#00263e").pack()
        self.drone_bar=ttk.Progressbar(left,length=180)
        self.drone_bar.pack(pady=3)

        tk.Label(left,text="Animal Probability",fg="#00ffcc",bg="#00263e").pack()
        self.animal_bar=ttk.Progressbar(left,length=180)
        self.animal_bar.pack(pady=3)

        tk.Label(left,text="Noise Probability",fg="#66ccff",bg="#00263e").pack()
        self.noise_bar=ttk.Progressbar(left,length=180)
        self.noise_bar.pack(pady=3)

        # CENTER TABLE

        center=tk.Frame(main,bg="#001a2c")
        center.grid(row=0,column=1,sticky="nsew")

        columns=("Time","Threat","Confidence","Location")

        self.tree=ttk.Treeview(center,columns=columns,show="headings",height=16)

        for col in columns:
            self.tree.heading(col,text=col)
            self.tree.column(col,anchor="center")

        self.tree.pack(fill="both",expand=True)

        # RIGHT PANEL

        right=tk.Frame(main,bg="#001a2c",width=260)
        right.grid(row=0,column=2,sticky="ns")

        tk.Label(right,text="TACTICAL RADAR",
                 fg="#00e6e6",bg="#001a2c",
                 font=("Consolas",12,"bold")).pack()

        self.canvas=tk.Canvas(right,width=220,height=220,bg="black")
        self.canvas.pack(pady=10)

        self.canvas.create_oval(20,20,200,200,outline="cyan")
        self.canvas.create_oval(70,70,150,150,outline="cyan")

        self.sweep=self.canvas.create_line(110,110,110,20,fill="cyan",width=2)

        tk.Label(right,text="THREAT CONFIDENCE GAUGE",
                 fg="red",bg="#001a2c",
                 font=("Consolas",12,"bold")).pack()

        self.gauge=tk.Canvas(right,width=220,height=120,bg="black")
        self.gauge.pack()

        self.gauge_arc=self.gauge.create_arc(
            10,10,210,200,start=180,extent=0,
            outline="yellow",width=4,style="arc"
        )

        tk.Label(right,text="THREAT LEVEL",
                 fg="red",bg="#001a2c",
                 font=("Consolas",12,"bold")).pack(pady=5)

        self.level=tk.Label(right,text="LOW",
                            bg="darkgreen",fg="#00ffcc",
                            font=("Consolas",18,"bold"),
                            width=10)
        self.level.pack()

    # ---------------------------------------------------

    def start_system(self):
        self.running=True
        self.status.config(text="STATUS: MONITORING")

    def stop_system(self):
        self.running=False
        self.status.config(text="STATUS: STOPPED")

    # ---------------------------------------------------

    def animate_radar(self):

        self.angle+=4
        if self.angle>360:
            self.angle=0

        x=110+90*math.cos(math.radians(self.angle))
        y=110+90*math.sin(math.radians(self.angle))

        self.canvas.coords(self.sweep,110,110,x,y)

        self.root.after(40,self.animate_radar)

    # ---------------------------------------------------

    def show_alert(self,label,location):

        alert=tk.Toplevel(self.root)
        alert.geometry("420x150+20+250")
        alert.configure(bg="#330000")
        alert.attributes("-topmost",True)

        tk.Label(alert,text="⚠ BORDER THREAT DETECTED",
                 fg="red",bg="#330000",
                 font=("Consolas",14,"bold")).pack(pady=10)

        tk.Label(alert,text=f"{label}\nLocation: {location}",
                 fg="white",bg="#330000",
                 font=("Consolas",12)).pack()

        alert.after(3500,alert.destroy)

    # ---------------------------------------------------

    def open_map(self,location):

        lat,lon=self.locations.get(location,[28.6139,77.2090])

        m = folium.Map(
            location=[lat,lon],
            zoom_start=7,
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri"
        )

        folium.Marker([lat,lon],
                      popup=f"Threat: {location}",
                      icon=folium.Icon(color="red")).add_to(m)

        path="threat_map.html"
        m.save(path)

        subprocess.Popen(["cmd","/c","start","/min",path])

    # ---------------------------------------------------

    def update_dashboard(self,label,confidence,probs):

        if not self.running:
            return
        
        

        location=random.choice(list(self.locations.keys()))

        self.gauge.itemconfig(self.gauge_arc,extent=0)

        self.total+=1

        if label=="Human Intrusion":
            self.human+=1
        elif label=="Drone Threat":
            self.drone+=1
        elif label=="Animal Threat":
            self.animal+=1
        else:
            self.noise+=1

        self.total_label.config(text=f"Total Predictions: {self.total}")
        self.human_label.config(text=f"Human Intrusion: {self.human}")
        self.drone_label.config(text=f"Drone Threat: {self.drone}")
        self.animal_label.config(text=f"Animal Threat: {self.animal}")
        self.noise_label.config(text=f"Environmental Noise: {self.noise}")

        self.human_bar["value"]=probs["Human Intrusion"]
        self.drone_bar["value"]=probs["Drone Threat"]
        self.animal_bar["value"]=probs["Animal Threat"]
        self.noise_bar["value"]=probs["Environmental Noise"]

        if label=="Human Intrusion":
            winsound.Beep(1500,300)

        if label=="Drone Threat":
            winsound.Beep(2000,300)

        if label in ["Human Intrusion","Drone Threat"]:

            self.show_alert(label,location)
            self.open_map(location)

            x,y=self.radar_positions.get(location,(110,110))

            dot=self.canvas.create_oval(x-4,y-4,x+4,y+4,fill="red")
            self.targets.append(dot)

            if len(self.targets)>3:
                self.canvas.delete(self.targets[0])
                self.targets.pop(0)

            extent=(confidence/100)*180
            self.gauge.itemconfig(self.gauge_arc,extent=-extent)

        # THREAT LEVEL

        if label in ["Human Intrusion","Drone Threat"]:
            self.level.config(text="HIGH",bg="red")

        elif label=="Animal Threat":
            self.level.config(text="MEDIUM",bg="orange")

        else:
            self.level.config(text="LOW",bg="darkgreen")

        t=time.strftime("%H:%M:%S")

        self.tree.insert("",0,values=(t,label,f"{confidence:.2f}%",location))