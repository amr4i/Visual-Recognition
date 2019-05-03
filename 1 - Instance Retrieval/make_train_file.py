import os
import csv

train_file = open("train.txt", "w")
train_file_obj = csv.writer(train_file)

bb_dict = {
	'aunt_jemima_original_syrup': {
		'N1':{
			'xmin': 270,
			'xmax': 355,		
			'ymin': 100,
			'ymax': 280,
		},
		'N2':{
			'xmin': 270,
			'xmax': 355,		
			'ymin': 70,
			'ymax': 230,
		}
	},
	'3m_high_tack_spray_adhesive': {
		'N1':{
			'xmin': 280,
			'xmax': 340,		
			'ymin': 110,
			'ymax': 280,
		},
		'N2':{
			'xmin': 280,
			'xmax': 340,		
			'ymin': 75,
			'ymax': 225,
		}
	},
	'campbells_chicken_noodle_soup': {
		'N1':{
			'xmin': 270,
			'xmax': 350,		
			'ymin': 205,
			'ymax': 280,
		},
		'N2':{
			'xmin': 270,
			'xmax': 350,		
			'ymin': 150,
			'ymax': 225,
		}
	},
	'cheez_it_white_cheddar': {
		'N1':{
			'xmin': 250,
			'xmax': 375,		
			'ymin': 140,
			'ymax': 270,
		},
		'N2':{
			'xmin': 250,
			'xmax': 370,		
			'ymin': 90,
			'ymax': 235,
		}
	},
	'cholula_chipotle_hot_sauce': {
		'N1':{
			'xmin': 295,
			'xmax': 330,		
			'ymin': 140,
			'ymax': 270,
		},
		'N2':{
			'xmin': 295,
			'xmax': 330,		
			'ymin': 105,
			'ymax': 220,
		}
	},
	'clif_crunch_chocolate_chip': {
		'N1':{
			'xmin': 250,
			'xmax': 365,		
			'ymin': 160,
			'ymax': 290,
		},
		'N2':{
			'xmin': 250,
			'xmax': 365,		
			'ymin': 115,
			'ymax': 230,
		}
	},
	'coca_cola_glass_bottle': {
		'N1':{
			'xmin': 285,
			'xmax': 335,		
			'ymin': 100,
			'ymax': 270,
		},
		'N2':{
			'xmin': 285,
			'xmax': 335,		
			'ymin': 70,
			'ymax': 225,
		}
	},
	'detergent': {
		'N1':{
			'xmin': 260,
			'xmax': 370,		
			'ymin': 180,
			'ymax': 285,
		},
		'N2':{
			'xmin': 265,
			'xmax': 355,		
			'ymin': 45,
			'ymax': 240,
		}
	},
	'expo_marker_red': {
		'N1':{
			'xmin': 300,
			'xmax': 325,		
			'ymin': 185,
			'ymax': 270,
		},
		'N2':{
			'xmin': 300,
			'xmax': 325,		
			'ymin': 150,
			'ymax': 215,
		}
	},
	'listerine_green': {
		'N1':{
			'xmin': 265,
			'xmax': 360,		
			'ymin': 85,
			'ymax': 270,
		},
		'N2':{
			'xmin': 265,
			'xmax': 360,		
			'ymin': 55,
			'ymax': 220,
		}
	},
	'nice_honey_roasted_almonds': {
		'N1':{
			'xmin': 270,
			'xmax': 350,		
			'ymin': 205,
			'ymax': 280,
		},
		'N2':{
			'xmin': 270,
			'xmax': 350,		
			'ymin': 150,
			'ymax': 230,
		}
	},
	'nutrigrain_apple_cinnamon': {
		'N1':{
			'xmin': 240,
			'xmax': 390,		
			'ymin': 160,
			'ymax': 280,
		},
		'N2':{
			'xmin': 230,
			'xmax': 380,		
			'ymin': 120,
			'ymax': 240,
		}
	},
	'palmolive_green': {
		'N1':{
			'xmin': 280,
			'xmax': 350,		
			'ymin': 140,
			'ymax': 270,
		},
		'N2':{
			'xmin': 280,
			'xmax': 350,		
			'ymin': 110,
			'ymax': 225,
		}
	},
	'pringles_bbq': {
		'N1':{
			'xmin': 280,
			'xmax': 340,		
			'ymin': 100,
			'ymax': 280,
		},
		'N2':{
			'xmin': 280,
			'xmax': 340,		
			'ymin': 60,
			'ymax': 220,
		}
	},
	'vo5_extra_body_volumizing_shampoo': {
		'N1':{
			'xmin': 280,
			'xmax': 340,		
			'ymin': 120,
			'ymax': 270,
		},
		'N2':{
			'xmin': 280,
			'xmax': 340,		
			'ymin': 90,
			'ymax': 220,
		}
	},
	'vo5_split_ends_anti_breakage_shampoo': {
		'N1':{
			'xmin': 280,
			'xmax': 340,		
			'ymin': 120,
			'ymax': 270,
		},
		'N2':{
			'xmin': 280,
			'xmax': 340,		
			'ymin': 90,
			'ymax': 220,
		}
	}
}

# train_file_obj.writerow(['image_names', 'class_type', 'xmin', 'xmax', 'ymin', 'ymax'])
for classname in os.listdir("../train"):
	for imageName in os.listdir("../train/"+classname):
		row = []
		imagePath = "../train/"+classname+"/"+imageName
		row.append(imagePath)
		row.append(bb_dict[classname][imageName.split('_')[0]]['xmin'])
		row.append(bb_dict[classname][imageName.split('_')[0]]['ymin'])
		row.append(bb_dict[classname][imageName.split('_')[0]]['xmax'])
		row.append(bb_dict[classname][imageName.split('_')[0]]['ymax'])
		row.append(classname)
		train_file_obj.writerow(row)

train_file.close()