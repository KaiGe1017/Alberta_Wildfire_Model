-- Add new column to store the frorest area.
ALTER TABLE Fact_Wildfire_2006_2023
ADD Forest_area CHAR(1);

-- Get the first letter in fire_number and insert to this new column.
UPDATE Fact_Wildfire_2006_2023
SET Forest_area = LEFT(fire_number, 1);

-- Check the new column is displayed correctly.
Select top(100) * FROM Fact_Wildfire_2006_2023

-- Create the Dimension tables from data dictionary.
-- Dim_forest_area
CREATE TABLE Dim_forest_area (
    forest_area CHAR(1) PRIMARY KEY,
    forest_area_detail VARCHAR(20)
);
INSERT INTO Dim_forest_area (forest_area, forest_area_detail)
VALUES 
('C', 'Calgary'),
('E', 'Edson'),
('H', 'High Level'),
('G', 'Grande Prairie'),
('L', 'Lac La Biche'),
('M', 'Fort McMurray'),
('P', 'Peace River'),
('R', 'Rocky'),
('S', 'Slave Lake'),
('W', 'Whitecourt');

-- Dim_size_class
CREATE TABLE Dim_size_class (
    size_class VARCHAR(10) PRIMARY KEY,
    size_class_datail VARCHAR(50)
);
INSERT INTO Dim_size_class (size_class, size_class_datail)
VALUES 
('A', '0.01-0.1 ha'),
('B', '0.11-1.0 ha'),
('C', '1.1-10.0 ha'),
('D', '10.1-100.0 ha'),
('E', '100.1-1000.0 ha'),
('F', '1000.1-5000.0 ha'),
('G', '5000.1+ ha');

-- Dim_det_agent
CREATE TABLE Dim_det_agent (
    det_agent VARCHAR(10) PRIMARY KEY,
    det_agent_detail VARCHAR(255)
);
INSERT INTO Dim_det_agent (det_agent, det_agent_detail)
VALUES 
('RAP', 'Rappel Crew'),
('HAC', 'Helitack Crew'),
('ASU', 'Unit Crew'),
('MD', 'Man Up Rotor Wing'),
('FW', 'Fixed Wing Patrol'),
('RW', 'Rotor Wing Patrol'),
('FRST', 'Forest Officer'),
('CREW', 'Wildfire Crew'),
('PATR', 'Patrolman'),
('PIND', 'Industry Patrol'),
('310', 'Called in on 310 FIRE phone line'),
('GOVT', 'Other Government Agencies'),
('LFS', 'Other Department Personnel'),
('PUB', 'General Public'),
('UAA', 'Unplanned Department Aircraft'),
('UIND', 'Unplanned Industry Aircraft'),
('UPA', 'Unplanned Public Aircraft'),
('AC', 'Atikaki'),
('AD', 'Adams Lake'),
('AF', 'Affleck'),
('AL', 'Albert Lake'),
('AM', 'Amisk'),
('AN', 'Anstey'),
('AR', 'Armstrong'),
('AT', 'Atlee'),
('AW', 'Awarua'),
('BB', 'Blind Bay'),
('BC', 'Buckton'),
('BR', 'Battle River'),
('BS', 'Bison Lake'),
('BT', 'Blackstone'),
('BZ', 'Brazeau'),
('BY', 'Baldy'),
('CA', 'Chinchaga'),
('CB', 'Carbondale'),
('CC', 'Carrot Creek'),
('CE', 'Cline'),
('CF', 'Cambrian'),
('CH', 'Clear Hills'),
('CK', 'Conklin'),
('CM', 'Chisholm'),
('CP', 'Cowpar Lake'),
('CT', 'Copton'),
('CU', 'Cadotte'),
('CY', 'Chipewyan Lakes'),
('DG', 'Doig'),
('DM', 'Deer Mountain'),
('DW', 'Deadwood'),
('EA', 'Eagle'),
('EC', 'Economy Creek'),
('ED', 'Edra'),
('EH', 'Enilda'),
('EL', 'Ells River'),
('FG', 'Foggy Mountain'),
('IM', 'Imperial'),
('IS', 'Ironstone'),
('JE', 'Jean Lake'),
('JO', 'Johnson Lake'),
('KA', 'Kakwa'),
('KE', 'Keg'),
('KK', 'Kananaskis'),
('KM', 'Kimiwan'),
('LG', 'Legend'),
('LI', 'Limestone'),
('LK', 'Livock'),
('LO', 'Lovett'),
('LV', 'Livingstone'),
('MB', 'Mayberne'),
('MH', 'Mockingbird Hill'),
('MN', 'Meridian'),
('MO', 'Moberly'),
('MR', 'Marten Mountain'),
('MS', 'Moose Mountain'),
('MQ', 'Muskwa'),
('MU', 'Muskeg Mountain'),
('MY', 'May'),
('NM', 'Nose Mountain'),
('NO', 'Notikewin'),
('OB', 'Obed'),
('OL', 'Otter Lakes'),
('SA', 'Saddle Hills'),
('SD', 'Swan Dive'),
('SG', 'Sugarloaf'),
('SI', 'Simonette'),
('SK', 'Smoky'),
('SN', 'Snuff Mountain'),
('SP', 'Salt Prairie'),
('SQ', 'Sandy Lake'),
('SR', 'Sand River'),
('ST', 'Stony Mountain'),
('SV', 'Steen'),
('SW', 'Sweathouse'),
('TM', 'Trout Mountain'),
('TO', 'Tom Hill'),
('TP', 'Teepee Lake'),
('TR', 'Torrens'),
('TT', 'Talbot Lake'),
('TY', 'Tony'),
('VG', 'Vega'),
('WC', 'Whitecourt'),
('WD', 'Whitemud'),
('WF', 'Whitefish'),
('WM', 'White Mountain'),
('WT', 'Watt Mountain'),
('WU', 'Wadlin'),
('YH', 'Yellowhead'),
('ZA', 'Zama');

-- Dim_det_agent_type
CREATE TABLE Dim_det_agent_type (
    det_agent_type VARCHAR(10) PRIMARY KEY,
    det_agent_type_detail VARCHAR(255)
);
INSERT INTO Dim_det_agent_type (det_agent_type, det_agent_type_detail)
VALUES 
('LKT', 'Lookout'),
('AIR', 'Air Patrol'),
('GRP', 'Ground Patrol'),
('UNP', 'Unplanned');

-- Dim_fuel_type
CREATE TABLE Dim_fuel_type (
    fuel_type VARCHAR(10) PRIMARY KEY,
    fuel_type_detail VARCHAR(255)
);
INSERT INTO Dim_fuel_type (fuel_type, fuel_type_detail)
VALUES 
('C1', 'Coniferous'),
('C2', 'Coniferous'),
('C3', 'Coniferous'),
('C4', 'Coniferous'),
('C7', 'Coniferous'),
('D1', 'Deciduous'),
('M1', 'Mixedwood'),
('M2', 'Mixedwood'),
('O1a', 'Grass'),
('O1b', 'Grass'),
('S1', 'Slash'),
('S2', 'Slash');
