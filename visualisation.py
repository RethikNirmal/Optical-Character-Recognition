import matplotlib.pyplot as plt
def visualize(data):
    
    #This one give us the number of digits in each image
    length = []
    for i in data:
        length.append(i['length'])
        
    plt.hist(length,color= ['red'])
    plt.xlabel('Sequence Length')
    plt.ylabel('No of images')
    