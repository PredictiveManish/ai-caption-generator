import time
start_time = time.time()
def dot_product_simple(list_a, list_b):
    """
    Calculates the dot product using standard pure Python loops.
    This method has higher latency for large inputs.
    """
    if len(list_a) != len(list_b):
        raise ValueError("Lists must be of the same length")
    
    result = 0
    
    for i in range(len(list_a)):
        result += list_a[i] * list_b[i]
    return result
end = time.time()

a = [1, 2, 3]
b = [4, 5, 6]
print(f"Simple Method Result: {dot_product_simple(a, b)}")
print(end-start_time)