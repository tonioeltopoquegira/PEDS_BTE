h=2;
Point(1) = {-50.0,50.0,0,h};
Point(2) = {50.0,50.0,0,h};
Point(3) = {50.0,-50.0,0,h};
Point(4) = {-50.0,-50.0,0,h};
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};
Line Loop(1) = {1,2,3,4};
Point(5) = {-35.0,35.0,0,h};
Point(6) = {-35.0,45.0,0,h};
Point(7) = {-45.0,45.0,0,h};
Point(8) = {-45.0,35.0,0,h};
Line(5) = {5,6};
Line(6) = {6,7};
Line(7) = {7,8};
Line(8) = {8,5};
Line Loop(2) = {5,6,7,8};
Plane Surface(1) = {1,2};
Physical Surface("Bulk") = {1};
Physical Line("Boundary") = {5,6,7,8};
Periodic Line{4} = {-2};
Physical Line("Periodic_x_a") = {4};
Physical Line("Periodic_x_b") = {2};
Periodic Line{3} = {-1};
Physical Line("Periodic_y_a") = {3};
Physical Line("Periodic_y_b") = {1};
