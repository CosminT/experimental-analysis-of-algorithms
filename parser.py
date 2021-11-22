class Parser:
    def __init__(self, file_path, file_path_solution):        
        self.speed_of_truck = None
        self.speed_of_drone = None
        self.number_of_nodes = None
        self.PATH = file_path
        self.PATH_SOL = file_path_solution
        self.X = []
        self.Y = []
        self.loc = []

    def check_numbers(self, el):
        try:
            float(el)
            return True
        except ValueError:
            return False

    def read_file(self, file_path):
        data = []
        file = open(file_path,"r")
        lines = file.readlines()
        for l in lines:
            l = l.replace("\n","")
            l = l.split(" ")
            if self.check_numbers(l[0]):
                data.append(l)
        return data

    def get_input_data(self):
        data = self.read_file(self.PATH)
        for el in range(len(data)):
            if el == 0:
                self.speed_of_truck = float(data[el][0])
            elif el == 1:
                self.speed_of_drone = float(data[el][0])
            elif el == 2:
                self.number_of_nodes = int(data[el][0])
            else:
                self.X.append(float(data[el][0]))
                self.Y.append(float(data[el][1]))
                self.loc.append(data[el][2])

