lc1 = 0.0125;
lc2 = 0.005;
h = 0.0001;

Point(1) = {0, 0, 0, lc1};
Point(2) = {1, 0, 0, lc1};
Point(3) = {0, 1, 0, lc1};
Point(4) = {1, 1, 0, lc1};

Point(5) = {0, 0.5-h, 0.0, lc1};
Point(6) = {0, 0.5+h, 0.0, lc1};
Point(7) = {0.5, 0.5, 0.0, lc2};
Point(8) = {1, 0.5, 0.0, lc2};

//+
Line(1) = {1, 2};
//+
Line(2) = {2, 8};
//+
Line(3) = {8, 7};
//+
Line(4) = {7, 5};
//+
Line(5) = {5, 1};
//+
Line(6) = {8, 4};
//+
Line(7) = {4, 3};
//+
Line(8) = {3, 6};
//+
Line(9) = {6, 7};
//+
Line Loop(1) = {5, 1, 2, 3, 4};
//+
Plane Surface(1) = {1};
//+
Line Loop(2) = {8, 9, -3, 6, 7};
//+
Plane Surface(2) = {2};
//+
Physical Surface(1) = {2, 1};
