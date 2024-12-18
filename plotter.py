
import matplotlib.pyplot as plt

class DynamicPlot:
    def __init__(self, title='Dynamic Plot', subplot_grid = (1, 1), set_size_inches=(15, 6)):
        plt.ion()  # Turn on interactive mode
        self.subplot_width, self.subplot_height = subplot_grid
        self.fig, self.axs = plt.subplots(*subplot_grid)
        # Set figure size
        self.fig.set_size_inches(*set_size_inches)
        
        plt.suptitle(title)
        
    def is_same_length(self, *args):
        length = len(args[0])
        for arg in args:
            if len(arg) != length:
                return False
        return True
    
    def save_plot_image(self, filename):
        plt.savefig(filename)

    def plot(self, **mult_data_info):
        mult_data = mult_data_info["mult_data"]
        titles = mult_data_info["titles"]
        xlabels = mult_data_info["xlabels"]
        ylabels = mult_data_info["ylabels"]
        
        if not self.is_same_length(mult_data, titles, xlabels, ylabels):
            print("Data length mismatch")
            return
        
        for n in range(len(mult_data)):
            i = n // self.subplot_width
            j = n % self.subplot_height
            
            if self.subplot_width == 1 or self.subplot_height == 1:
                index = n
            else:
                index = (i, j)
            
            self.axs[index].clear()
            self.axs[index].plot(mult_data[n], marker='')
            self.axs[index].set_title(titles[n])
            self.axs[index].set_xlabel(xlabels[n])
            self.axs[index].set_ylabel(ylabels[n])
            
        plt.draw()
        plt.pause(0.01)
        
if __name__ == "__main__":
    # Example usage:
    plotter = DynamicPlot(title="Test Plot", subplot_grid=(2, 2), set_size_inches=(10, 10))
    
    import time
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    
    for i in range(100):
        x1.append(i**(1/2))
        x2.append(i**2)
        x3.append(i**(1/3))
        x4.append(i**3)

        
        data_info = {
            "mult_data": [x1, x2, x3, x4],
            "titles": ["x1", "x2", "x3", "x4"],
            "xlabels": ["x"]*4, 
            "ylabels": ["y"]*4
        }
        
        plotter.plot(**data_info)
        
        time.sleep(0.1)