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
Point(9) = {-25.0,35.0,0,h};
Point(10) = {-15.0,35.0,0,h};
Point(11) = {-15.0,45.0,0,h};
Point(12) = {-25.0,45.0,0,h};
Line(9) = {9,10};
Line(10) = {10,11};
Line(11) = {11,12};
Line(12) = {12,9};
Line Loop(3) = {9,10,11,12};
Point(13) = {-5.0,45.0,0,h};
Point(14) = {-5.0,35.0,0,h};
Point(15) = {5.0,35.0,0,h};
Point(16) = {5.0,45.0,0,h};
Line(13) = {13,14};
Line(14) = {14,15};
Line(15) = {15,16};
Line(16) = {16,13};
Line Loop(4) = {13,14,15,16};
Point(17) = {25.0,45.0,0,h};
Point(18) = {15.0,45.0,0,h};
Point(19) = {15.0,35.0,0,h};
Point(20) = {25.0,35.0,0,h};
Line(17) = {17,18};
Line(18) = {18,19};
Line(19) = {19,20};
Line(20) = {20,17};
Line Loop(5) = {17,18,19,20};
Point(21) = {45.0,35.0,0,h};
Point(22) = {45.0,45.0,0,h};
Point(23) = {35.0,45.0,0,h};
Point(24) = {35.0,35.0,0,h};
Line(21) = {21,22};
Line(22) = {22,23};
Line(23) = {23,24};
Line(24) = {24,21};
Line Loop(6) = {21,22,23,24};
Point(25) = {-45.0,15.0,0,h};
Point(26) = {-35.0,15.0,0,h};
Point(27) = {-35.0,25.0,0,h};
Point(28) = {-45.0,25.0,0,h};
Line(25) = {25,26};
Line(26) = {26,27};
Line(27) = {27,28};
Line(28) = {28,25};
Line Loop(7) = {25,26,27,28};
Point(29) = {-25.0,25.0,0,h};
Point(30) = {-25.0,15.0,0,h};
Point(31) = {-15.0,15.0,0,h};
Point(32) = {-15.0,25.0,0,h};
Line(29) = {29,30};
Line(30) = {30,31};
Line(31) = {31,32};
Line(32) = {32,29};
Line Loop(8) = {29,30,31,32};
Point(33) = {5.0,25.0,0,h};
Point(34) = {-5.0,25.0,0,h};
Point(35) = {-5.0,15.0,0,h};
Point(36) = {5.0,15.0,0,h};
Line(33) = {33,34};
Line(34) = {34,35};
Line(35) = {35,36};
Line(36) = {36,33};
Line Loop(9) = {33,34,35,36};
Point(37) = {25.0,15.0,0,h};
Point(38) = {25.0,25.0,0,h};
Point(39) = {15.0,25.0,0,h};
Point(40) = {15.0,15.0,0,h};
Line(37) = {37,38};
Line(38) = {38,39};
Line(39) = {39,40};
Line(40) = {40,37};
Line Loop(10) = {37,38,39,40};
Point(41) = {35.0,15.0,0,h};
Point(42) = {45.0,15.0,0,h};
Point(43) = {45.0,25.0,0,h};
Point(44) = {35.0,25.0,0,h};
Line(41) = {41,42};
Line(42) = {42,43};
Line(43) = {43,44};
Line(44) = {44,41};
Line Loop(11) = {41,42,43,44};
Point(45) = {-45.0,5.0,0,h};
Point(46) = {-45.0,-5.0,0,h};
Point(47) = {-35.0,-5.0,0,h};
Point(48) = {-35.0,5.0,0,h};
Line(45) = {45,46};
Line(46) = {46,47};
Line(47) = {47,48};
Line(48) = {48,45};
Line Loop(12) = {45,46,47,48};
Point(49) = {-15.0,5.0,0,h};
Point(50) = {-25.0,5.0,0,h};
Point(51) = {-25.0,-5.0,0,h};
Point(52) = {-15.0,-5.0,0,h};
Line(49) = {49,50};
Line(50) = {50,51};
Line(51) = {51,52};
Line(52) = {52,49};
Line Loop(13) = {49,50,51,52};
Point(53) = {5.0,-5.0,0,h};
Point(54) = {5.0,5.0,0,h};
Point(55) = {-5.0,5.0,0,h};
Point(56) = {-5.0,-5.0,0,h};
Line(53) = {53,54};
Line(54) = {54,55};
Line(55) = {55,56};
Line(56) = {56,53};
Line Loop(14) = {53,54,55,56};
Point(57) = {15.0,-5.0,0,h};
Point(58) = {25.0,-5.0,0,h};
Point(59) = {25.0,5.0,0,h};
Point(60) = {15.0,5.0,0,h};
Line(57) = {57,58};
Line(58) = {58,59};
Line(59) = {59,60};
Line(60) = {60,57};
Line Loop(15) = {57,58,59,60};
Point(61) = {35.0,5.0,0,h};
Point(62) = {35.0,-5.0,0,h};
Point(63) = {45.0,-5.0,0,h};
Point(64) = {45.0,5.0,0,h};
Line(61) = {61,62};
Line(62) = {62,63};
Line(63) = {63,64};
Line(64) = {64,61};
Line Loop(16) = {61,62,63,64};
Point(65) = {-35.0,-15.0,0,h};
Point(66) = {-45.0,-15.0,0,h};
Point(67) = {-45.0,-25.0,0,h};
Point(68) = {-35.0,-25.0,0,h};
Line(65) = {65,66};
Line(66) = {66,67};
Line(67) = {67,68};
Line(68) = {68,65};
Line Loop(17) = {65,66,67,68};
Point(69) = {-15.0,-25.0,0,h};
Point(70) = {-15.0,-15.0,0,h};
Point(71) = {-25.0,-15.0,0,h};
Point(72) = {-25.0,-25.0,0,h};
Line(69) = {69,70};
Line(70) = {70,71};
Line(71) = {71,72};
Line(72) = {72,69};
Line Loop(18) = {69,70,71,72};
Point(73) = {-5.0,-25.0,0,h};
Point(74) = {5.0,-25.0,0,h};
Point(75) = {5.0,-15.0,0,h};
Point(76) = {-5.0,-15.0,0,h};
Line(73) = {73,74};
Line(74) = {74,75};
Line(75) = {75,76};
Line(76) = {76,73};
Line Loop(19) = {73,74,75,76};
Point(77) = {15.0,-15.0,0,h};
Point(78) = {15.0,-25.0,0,h};
Point(79) = {25.0,-25.0,0,h};
Point(80) = {25.0,-15.0,0,h};
Line(77) = {77,78};
Line(78) = {78,79};
Line(79) = {79,80};
Line(80) = {80,77};
Line Loop(20) = {77,78,79,80};
Point(81) = {45.0,-15.0,0,h};
Point(82) = {35.0,-15.0,0,h};
Point(83) = {35.0,-25.0,0,h};
Point(84) = {45.0,-25.0,0,h};
Line(81) = {81,82};
Line(82) = {82,83};
Line(83) = {83,84};
Line(84) = {84,81};
Line Loop(21) = {81,82,83,84};
Point(85) = {-35.0,-45.0,0,h};
Point(86) = {-35.0,-35.0,0,h};
Point(87) = {-45.0,-35.0,0,h};
Point(88) = {-45.0,-45.0,0,h};
Line(85) = {85,86};
Line(86) = {86,87};
Line(87) = {87,88};
Line(88) = {88,85};
Line Loop(22) = {85,86,87,88};
Point(89) = {-25.0,-45.0,0,h};
Point(90) = {-15.0,-45.0,0,h};
Point(91) = {-15.0,-35.0,0,h};
Point(92) = {-25.0,-35.0,0,h};
Line(89) = {89,90};
Line(90) = {90,91};
Line(91) = {91,92};
Line(92) = {92,89};
Line Loop(23) = {89,90,91,92};
Point(93) = {-5.0,-35.0,0,h};
Point(94) = {-5.0,-45.0,0,h};
Point(95) = {5.0,-45.0,0,h};
Point(96) = {5.0,-35.0,0,h};
Line(93) = {93,94};
Line(94) = {94,95};
Line(95) = {95,96};
Line(96) = {96,93};
Line Loop(24) = {93,94,95,96};
Point(97) = {25.0,-35.0,0,h};
Point(98) = {15.0,-35.0,0,h};
Point(99) = {15.0,-45.0,0,h};
Point(100) = {25.0,-45.0,0,h};
Line(97) = {97,98};
Line(98) = {98,99};
Line(99) = {99,100};
Line(100) = {100,97};
Line Loop(25) = {97,98,99,100};
Point(101) = {45.0,-45.0,0,h};
Point(102) = {45.0,-35.0,0,h};
Point(103) = {35.0,-35.0,0,h};
Point(104) = {35.0,-45.0,0,h};
Line(101) = {101,102};
Line(102) = {102,103};
Line(103) = {103,104};
Line(104) = {104,101};
Line Loop(26) = {101,102,103,104};
Plane Surface(1) = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26};
Physical Surface("Bulk") = {1};
Physical Line("Boundary") = {5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104};
Periodic Line{4} = {-2};
Physical Line("Periodic_x_a") = {4};
Physical Line("Periodic_x_b") = {2};
Periodic Line{3} = {-1};
Physical Line("Periodic_y_a") = {3};
Physical Line("Periodic_y_b") = {1};
