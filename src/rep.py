
def getRep(image, net):
    if(image is None):
        return None
    return net.forward(image)

def getRepString(image, net):
    rep = getRep(image, net)
    output = ""
    i = 0
    while(i < len(rep) - 1):
        output += str(rep[i]) + ","
        i += 1
    output += str(rep[i])
    return output
    