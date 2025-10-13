using StaticArrays: SA
function fib(n::UInt)
    n_bits = 8 * sizeof(UInt) - leading_zeros(n)
    A = SA{BigInt}[1 1; 1 0]
    B = SA{BigInt}[1; 0]
    for i in 0 : n_bits - 1
        B = iszero((n >> i) & 0b1) ? B : A * B
        A = A^2
    end
    return B[1]
end 
