coef_info = { 
  coef = function(x, y, z)
    local x2 = x
    local y2 = y
    local z2 = z
    for i = 1, 100 do
      x2 = x2 + i
      y2 = y2 + (2 * i)
      z2 = z2 + (3 * i)
    end
    -- cross product
    local comp0 = (y * z2) - (z * y2)
    local comp1 = (x * z2) - (z * x2)
    local comp2 = (x * y2) - (y * z2)
    return comp0 + comp1 + comp2
  end,
  component = 1,
}
