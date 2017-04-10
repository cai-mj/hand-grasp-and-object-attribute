#include <opencv2/opencv.hpp>

#include <string>
#include <vector>
#include <fstream>

using namespace std;
using namespace cv;

void transYamlfromOpencv2Pyyaml()
{
	string baseDir("/home/cai-mj/_GTA/annotation/GTEA");
	char command[200];
	vector<string> seqs;
/*	seqs.push_back("002");
	seqs.push_back("003");
	seqs.push_back("005");
	seqs.push_back("006");
	seqs.push_back("007");
	seqs.push_back("008");
	seqs.push_back("010");
	seqs.push_back("012");
	seqs.push_back("013");
	seqs.push_back("014");
	seqs.push_back("016");
	seqs.push_back("017");
	seqs.push_back("018");
	seqs.push_back("020");
	seqs.push_back("021");
	seqs.push_back("022");*/
	seqs.push_back("Alireza_American");
	seqs.push_back("Alireza_Snack");

	for(int i = 0; i < (int)seqs.size(); i++)
	{
		command[0] = 0;
		sprintf(command, "ls %s/%s/*.yml > filename.txt", baseDir.c_str(), seqs[i].c_str());
		system(command);

		fstream fs("filename.txt", ios::in);
		string val;
		vector<string> files;
		while(fs>>val) files.push_back(val);
		fs.close();

		for(int j = 0; j < (int)files.size(); j++)
		{
			FileStorage ymlfile(files[j], FileStorage::READ);
		    if(!ymlfile.isOpened())
		    {
		        cout << "ERROR: bugs of getting yml files: " << files[j] << endl;
				exit(1);
		    }

		    //read value from yaml file...
		    string seqname;
			string filename;
			int width;
			int height;
			int depth;
			string hLgrasp;
			int hLvisible;
			int hLminx;
			int hLminy;
			int hLmaxx;
			int hLmaxy;
			string hRgrasp;
			int hRvisible;
			int hRminx;
			int hRminy;
			int hRmaxx;
			int hRmaxy;
			string oLname;
			int oLvisible;
			int oLminx;
			int oLminy;
			int oLmaxx;
			int oLmaxy;
			bool isLprismatic;
			bool isLsphere;
			bool isLflat;
			bool isLrigid;
			string oRname;
			int oRvisible;
			int oRminx;
			int oRminy;
			int oRmaxx;
			int oRmaxy;
			bool isRprismatic;
			bool isRsphere;
			bool isRflat;
			bool isRrigid;
		    ymlfile["seqname"] >> seqname;
			ymlfile["filename"] >> filename;
			ymlfile["size"]["width"] >> width;
			ymlfile["size"]["height"] >> height;
			ymlfile["size"]["depth"] >> depth;
//			if((int)fs["lefthand"]["visible"])
		    {
		    	ymlfile["lefthand"]["visible"] >> hLvisible;
		        ymlfile["lefthand"]["grasp"] >> hLgrasp;
		        ymlfile["lefthand"]["bndbox"]["xmin"] >> hLminx;
		        ymlfile["lefthand"]["bndbox"]["ymin"] >> hLminy;
		        ymlfile["lefthand"]["bndbox"]["xmax"] >> hLmaxx;
		        ymlfile["lefthand"]["bndbox"]["ymax"] >> hLmaxy;
		    }
//		    if((int)fs["righthand"]["visible"])
		    {
		    	ymlfile["righthand"]["visible"] >> hRvisible;
		        ymlfile["righthand"]["grasp"] >> hRgrasp;
		        ymlfile["righthand"]["bndbox"]["xmin"] >> hRminx;
		        ymlfile["righthand"]["bndbox"]["ymin"] >> hRminy;
		        ymlfile["righthand"]["bndbox"]["xmax"] >> hRmaxx;
		        ymlfile["righthand"]["bndbox"]["ymax"] >> hRmaxy;
		    }
//		    if((int)fs["leftobject"]["visible"])
		    {
		    	ymlfile["leftobject"]["visible"] >> oLvisible;
		    	ymlfile["leftobject"]["name"] >> oLname;
		        ymlfile["leftobject"]["bndbox"]["xmin"] >> oLminx;
		        ymlfile["leftobject"]["bndbox"]["ymin"] >> oLminy;
		        ymlfile["leftobject"]["bndbox"]["xmax"] >> oLmaxx;
		        ymlfile["leftobject"]["bndbox"]["ymax"] >> oLmaxy;
		        ymlfile["leftobject"]["attribute"]["prismatic"] >> isLprismatic;
		        ymlfile["leftobject"]["attribute"]["sphere"] >> isLsphere;
		        ymlfile["leftobject"]["attribute"]["flat"] >> isLflat;
		        ymlfile["leftobject"]["attribute"]["rigid"] >> isLrigid;
		    }
//		    if((int)fs["rightobject"]["visible"])
		    {
		    	ymlfile["rightobject"]["visible"] >> oRvisible;
		    	ymlfile["rightobject"]["name"] >> oRname;
		        ymlfile["rightobject"]["bndbox"]["xmin"] >> oRminx;
		        ymlfile["rightobject"]["bndbox"]["ymin"] >> oRminy;
		        ymlfile["rightobject"]["bndbox"]["xmax"] >> oRmaxx;
		        ymlfile["rightobject"]["bndbox"]["ymax"] >> oRmaxy;
		        ymlfile["rightobject"]["attribute"]["prismatic"] >> isRprismatic;
		        ymlfile["rightobject"]["attribute"]["sphere"] >> isRsphere;
		        ymlfile["rightobject"]["attribute"]["flat"] >> isRflat;
		        ymlfile["rightobject"]["attribute"]["rigid"] >> isRrigid;
		    }
			ymlfile.release();

			fstream ofs(files[j], ios::out);
//			ofs << "% YAML: 1.0\n";
			ofs << "seqname: !!str " << seqname << endl;
			ofs << "filename: !!str " << filename << endl;
			ofs << "annotator: Minjie Cai\n";
			ofs << "size: { width: " << width << ", height: " << height << ", depth: " << depth << " }\n";
		    ofs << "lefthand: \n";
			ofs << "   visible: " << hLvisible << endl;
			ofs << "   grasp: " << hLgrasp << endl;
			ofs << "   bndbox: " << "{ xmin: " << hLminx << ", ymin: " << hLminy << ", xmax: " << hLmaxx << ", ymax: " << hLmaxy << " }\n";
			ofs << "righthand: \n";
			ofs << "   visible: " << hRvisible << endl;
			ofs << "   grasp: " << hRgrasp << endl;
			ofs << "   bndbox: " << "{ xmin: " << hRminx << ", ymin: " << hRminy << ", xmax: " << hRmaxx << ", ymax: " << hRmaxy << " }\n";
			ofs << "leftobject: \n";
			ofs << "   visible: " << oLvisible << endl;
			ofs << "   name: " << oLname << endl;
			ofs << "   bndbox: " << "{ xmin: " << oLminx << ", ymin: " << oLminy << ", xmax: " << oLmaxx << ", ymax: " << oLmaxy << " }\n";
			ofs << "   attribute: " << "{ prismatic: " << isLprismatic << ", sphere: " << isLsphere << ", flat: " << isLflat << ", rigid: " << isLrigid << " }\n";
			ofs << "rightobject: \n";
			ofs << "   visible: " << oRvisible << endl;
			ofs << "   name: " << oRname << endl;
			ofs << "   bndbox: " << "{ xmin: " << oRminx << ", ymin: " << oRminy << ", xmax: " << oRmaxx << ", ymax: " << oRmaxy << " }\n";
			ofs << "   attribute: " << "{ prismatic: " << isRprismatic << ", sphere: " << isRsphere << ", flat: " << isRflat << ", rigid: " << isRrigid << " }\n";
			ofs.close();
		}
	}
		
}

int main(int argc, char** agrv)
{
	transYamlfromOpencv2Pyyaml();
	return 0;
}
