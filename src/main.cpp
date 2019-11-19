#include <igl/readPLY.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/writePLY.h>
#include <igl/file_exists.h>
#include <imgui/imgui.h>
#include <iostream>
#include <random>
#include "mytools.h"
#include <Eigen/SVD>

#include <algorithm>
#include <vector>
#include <ANN/ANN.h>
#include <unistd.h>


class MyContext
{
public:

	std::string sourceFile1 = "../../data/SPRING0001.obj";


	MyContext() : point_size(2), line_width(0.5), mode(0)//, model(0)
	{

        std::cout<<"eigen version.:"<<EIGEN_WORLD_VERSION<<"."<<EIGEN_MAJOR_VERSION <<"."<< EIGEN_MINOR_VERSION<<"\n";
        char *path=NULL;
        size_t size;
        path=getcwd(path,size);
        std::cout<<"\n current Path "<<path << std::endl;
        //CLION



	}
	~MyContext() {}
    // front
    int mode;
    float point_size;
    float line_width;

    bool reduction = true;
    int reductionSize=4;
    int nModels = 25;
    int maxModel;
    int model = 1;
    int k_eig = 10;

    float scaling0 = 1;
    float scaling1 = 1;
    float scaling2 = 1;
    float scaling3 = 1;
    float scaling4 = 1;
    float scaling5 = 1;
    float threshold = 0.01;

    // back
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
    Eigen::MatrixXf eigenvectors;
    Eigen::MatrixXf evalues;
    Eigen::MatrixXf S,full_S,U;
    Eigen::VectorXf s_mean;
    Eigen::MatrixXd sampled_Points;

    std::string path = "../../data/";
    std::string meshName = "SPRING";
    std::string suffix = ".obj";
    std::string number;
    std::string filename;
    int needed_zero_length = 4;

    ANNpointArray getANNpointsFromMatrix(Eigen::MatrixXd points) {
        ANNpointArray P = annAllocPts(points.rows(), points.cols());
        for (int row = 0; row < points.rows(); row++) {
            ANNpoint point = annAllocPt(points.cols());
            for (int col = 0; col < points.cols(); col++) {
                point[col] = points(row, col);
            }
            P[row] = point;
        }
        return P;
    }

	void reset_display(igl::opengl::glfw::Viewer& viewer)
	{
		static std::default_random_engine generator;
		viewer.data().clear();
        viewer.data().show_lines = 0;
        viewer.data().show_overlay_depth = 1;


		if (mode == 1)
		{
            print("Start loading meshes ");
            // data folder contains SPRINGxxxx.obj files
            // some numbers are missing so we will retrieve only the first nModels models
            int column = 1;
            int i = 1;
            while(column <= nModels) {
                // create file path of the mesh model to load
                number = pad_number(i, needed_zero_length);
                filename = path + meshName + number + suffix;

                if (igl::file_exists(filename)) {
                    // read mesh
                    std::cout << filename << std::endl;
                    igl::readOBJ(filename, V, F);

                    int nRows = V.rows();
                    Eigen::MatrixXf sampledMesh;
                    sampledMesh = downsample(V,false,reductionSize,nRows);
                    sampledMesh.resize(3 * nRows, 1);
                    full_S.conservativeResize(3 * nRows, column);
                    full_S.col(column - 1) = sampledMesh;

                    if(reduction){
                        // optionally subsample only a set of points
                        sampledMesh = downsample(V, reduction, reductionSize, nRows);
                        sampledMesh.resize(3 * nRows, 1);
                        // S will be out matrix 3n x k, (n=#vertices, k=#models) containing all the models
                        S.conservativeResize(3 * nRows, column);
                        S.col(column - 1) = sampledMesh;
                    }

                    column++;
                }
                i++;
            }
            print(" mesh loaded");
            print("S matrix built , S is 3n x k , where n is # of vertices and k is # of models");

            // compute mean shape
            s_mean = S.rowwise().mean();
            std::cout << "mean shape computed from S, size: " << s_mean.rows() << " by " << s_mean.cols() << std::endl;
            // s_mean has size 3nx1, we multiply for a row vector 1xk of ones to repeat it
            // U has same size as S
            U = S - s_mean * Eigen::RowVectorXf::Ones(S.cols());
            std::cout << "U matrix (S - mean shape), size: " << U.rows() << " by " << U.cols() << std::endl;
            // decomposition to find the eigenvectors
            decomp( U , k_eig, eigenvectors, evalues);

            maxModel = nModels;

		}
		else if (mode == 2)
		{
		    print("Showing mean shape");

            Eigen::MatrixXd outputMean = recompose_V_matrix(s_mean);

            if(reduction){
                viewer.data().add_points(outputMean,Eigen::RowVector3d(0, 0, 0));
            } else {
                viewer.data().set_mesh(V, F);
                viewer.core.align_camera_center(V, F);
            }

            viewer.data().show_overlay_depth = 1;
            viewer.data().show_overlay = 1;
		}
		else if(mode == 3)
		{
		    print("Start PCA on mean mesh");

            Eigen::MatrixXf outputMesh(s_mean.rows(),s_mean.cols());
            outputMesh.setZero();
            // reconstruct
            Eigen::RowVectorXf scalar =  Eigen::RowVectorXf::Ones(eigenvectors.cols());
            scalar(0) = scaling0;
            scalar(1) = scaling1;
            scalar(2) = scaling2;
            scalar(3) = scaling3;
            scalar(4) = scaling4;
            scalar(5) = scaling5;
            int maxCols = eigenvectors.cols();
            for (int frequency = 0; frequency < eigenvectors.cols(); frequency++) {

                Eigen::MatrixXf a;
                a = s_mean.transpose() * eigenvectors.col(frequency);
                outputMesh.col(0) += a(0)* scalar[frequency] *eigenvectors.col(frequency);

            }

            sampled_Points = recompose_V_matrix(outputMesh.col(0));

            viewer.data().add_points(sampled_Points,Eigen::RowVector3d(0, 0, 0));
            //viewer.data().set_mesh(V, F);
            //viewer.core.align_camera_center(V, F);
            viewer.data().show_overlay_depth = 1;
            viewer.data().show_overlay = 1;
            /*
		}
        else if(mode == 4)
        {
             */
            print("starting upsampling ");
            Eigen::MatrixXd sampled_mean_shape = recompose_V_matrix(s_mean);
            Eigen::MatrixXd full_mean_shape = recompose_V_matrix(full_S.rowwise().mean());
            //std::cout <<"sampled_mean_shape" << sampled_mean_shape.rows() << " " << sampled_mean_shape.cols() << std::endl;
            std::cout <<"full_mean_shape" << full_mean_shape.rows() << " " << full_mean_shape.cols() << std::endl;
            //Eigen::MatrixXd IdxNNs = findIdxOfNN(full_mean_shape,sampled_mean_shape,4);
            //std::cout <<"IdxNNs " << IdxNNs.rows() << " " << IdxNNs.cols() << std::endl;
            Eigen::MatrixXd transformed_sampled_points = sampled_Points;
            std::cout <<"transformed_sampled_points" << transformed_sampled_points.rows() << " " << transformed_sampled_points.cols() << std::endl;

            ANNpointArray PANN = getANNpointsFromMatrix(sampled_mean_shape);
            ANNpointArray queryPoints = getANNpointsFromMatrix(full_mean_shape);
            ANNkd_tree PkdTree = ANNkd_tree(PANN, sampled_mean_shape.rows(), sampled_mean_shape.cols());
            Eigen::MatrixXd upsampledMesh = Eigen::MatrixXd::Zero(full_mean_shape.rows(), full_mean_shape.cols());
            print("start loop");
            for (int i = 0; i < full_mean_shape.rows(); i++) {
                if (i % reductionSize > 0) {

                    ANNidxArray neighbourIndicies = new ANNidx[4];
                    ANNdistArray neighbourDistances = new ANNdist[4];
                    ANNpoint queryPoint = queryPoints[i];

                    PkdTree.annkSearch(queryPoint, 4, neighbourIndicies, neighbourDistances, 0.0);

                    // find barycenter of queryPoint based on neighbours
                    Eigen::Vector3d q = full_mean_shape.row(i);

                    Eigen::Vector3d p1 = sampled_mean_shape.row(neighbourIndicies[0]);
                    Eigen::Vector3d p2 = sampled_mean_shape.row(neighbourIndicies[1]);
                    Eigen::Vector3d p3 = sampled_mean_shape.row(neighbourIndicies[2]);
                    Eigen::Vector3d p4 = sampled_mean_shape.row(neighbourIndicies[3]);
                    // solve Ax=b system
                    Eigen::Matrix4d A;
                    A << p1.row(0),p2.row(0),p3.row(0),p4.row(0), p1.row(1),p2.row(1),p3.row(1),p4.row(1), p1.row(2),p2.row(2),p3.row(2),p4.row(2), 1,1,1,1;
                    Eigen::Vector4d b;
                    b << q.row(0),q.row(1),q.row(2),1;

                    Eigen::Vector4d x = A.colPivHouseholderQr().solve(b);

                    // get neighbours in the updated_sampled_mean_points array

                    Eigen::Vector3d np1 = transformed_sampled_points.row(neighbourIndicies[0]);
                    Eigen::Vector3d np2 = transformed_sampled_points.row(neighbourIndicies[1]);
                    Eigen::Vector3d np3 = transformed_sampled_points.row(neighbourIndicies[2]);
                    Eigen::Vector3d np4 = transformed_sampled_points.row(neighbourIndicies[3]);

                    // get queryPoint position from updated_sampled_mean_points barycenter position
                    Eigen::Vector3d updated_position = (np1 * x.row(0)) + (np2 * x.row(1)) + (np3 * x.row(2)) + (np4 * x.row(3));

                    float distance = (np1-updated_position).norm();
                    if (distance > threshold){
                        updated_position = (np1+np2+np3+np4)/4;
                    }

                    // add this position to the upsampled mesh
                    upsampledMesh.row(i) = updated_position;
                    delete[] neighbourIndicies;
                } else {
                    int index = i / reductionSize;
                    upsampledMesh.row(i) = transformed_sampled_points.row(index);
                }
            }
            // visualize
            viewer.data().add_points(transformed_sampled_points,Eigen::RowVector3d(1, 1, 1));
            viewer.data().add_points(upsampledMesh,Eigen::RowVector3d(0, 0, 0));
            viewer.data().set_mesh(upsampledMesh,F);
            viewer.data().show_overlay_depth = 1;
            viewer.data().show_overlay = 1;
        }
        else if(mode == 5)
        {
            print(" start PCA with target model ");

            Eigen::MatrixXf targetV = S.col(model-1);

            Eigen::MatrixXf outputMesh(s_mean.rows(),s_mean.cols());
            outputMesh.setZero();
            Eigen::RowVectorXf scalar =  Eigen::RowVectorXf::Ones(eigenvectors.cols());
            scalar(0) = scaling0;
            scalar(1) = scaling1;
            scalar(2) = scaling2;
            scalar(3) = scaling3;
            scalar(4) = scaling4;
            scalar(5) = scaling5;
            int maxCols = eigenvectors.cols();
            for (int frequency = 0; frequency < eigenvectors.cols(); frequency++) {

                Eigen::MatrixXf a;
                a = (targetV.transpose() - s_mean.transpose()) * eigenvectors.col(frequency);
                outputMesh.col(0) += a(0)* scalar(frequency) *eigenvectors.col(frequency);

            }
            outputMesh.col(0) += s_mean;

            V = recompose_V_matrix(outputMesh.col(0));

            viewer.data().add_points(V,Eigen::RowVector3d(0, 0, 0));
            //viewer.data().set_mesh(V, F);
            //viewer.core.align_camera_center(V, F);
            viewer.data().show_overlay_depth = 1;
            viewer.data().show_overlay = 1;

        }
        else if(mode == 7)
        {
            print(" testing PCA with a single mesh ");
            Eigen::MatrixXf testV = V.cast<float>();
            int maxCols = eigenvectors.cols();
            Eigen::MatrixXf output;
            output.resize(testV.rows(),testV.cols());
            output.setZero();

            Eigen::RowVectorXf scalar =  Eigen::RowVectorXf::Ones(eigenvectors.cols());
            scalar(0) = scaling0;
            scalar(1) = scaling1;
            scalar(2) = scaling2;
            scalar(3) = scaling3;
            scalar(4) = scaling4;
            scalar(5) = scaling5;

            for (int frequency = 0; frequency < eigenvectors.cols(); frequency++) {

                Eigen::MatrixXf a;
                a = testV.transpose()* eigenvectors.col(frequency);

                output.col(0) += a(0)* scalar(0) *eigenvectors.col(frequency);
                output.col(1) += a(1)* scalar(1) *eigenvectors.col(frequency);
                output.col(2) += a(2)* scalar(2) *eigenvectors.col(frequency);
            }
            Eigen::MatrixXd showMesh;
            showMesh = output.cast<double>();
            viewer.data().set_mesh(showMesh,F);

            viewer.data().show_overlay_depth = 1;
            viewer.data().show_overlay = 1;
        }
        else if(mode == 8)
        {
            print(" testing with a single mesh ");
            std::string filename = "../../data/SPRING0001.obj";
            igl::readPLY(filename, V, F);

            Eigen::MatrixXf testV = V.cast<float>();
            Eigen::MatrixXf testMean = testV.colwise().mean();
            std::cout << "testMean "<< testMean.rows() << " " << testMean.cols() << std::endl;
            testV = testV - Eigen::RowVectorXf::Ones(testV.rows()).transpose()*testMean;

            decomp(testV,k_eig,eigenvectors,evalues);

            Eigen::MatrixXd outputV = testV.cast<double>();

            viewer.data().set_mesh(outputV,F);
            viewer.core.align_camera_center(outputV, F);
            viewer.data().set_face_based(true);
            viewer.data().show_overlay_depth = 1;
            viewer.data().show_overlay = 1;

        }
		else {
			print("Resetting to single mesh");

			Eigen::MatrixXf a(3,3);
			a<< 1,2,3,4,5,6,7,8,9;
			std::cout << a << std::endl;
			a.transpose().resize(9,1);
            std::cout << a << std::endl;


            switch(model){
                default:
                    sourceFile1 = "../../data/SPRING0001.obj";
                    break;
                case 1:
                    sourceFile1 = "../../data/SPRING0002.obj";
                    break;
            }

			igl::readOBJ(sourceFile1, V, F);
            viewer.data().set_mesh(V, F);
			viewer.core.align_camera_center(V, F);
			viewer.data().show_overlay_depth = 1;
			viewer.data().show_overlay = 1;
		}

		viewer.data().line_width = line_width;
		viewer.data().point_size = point_size;

	}

private:

};

MyContext g_myctx;


bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{

	std::cout << "Key: " << key << " " << (unsigned int)key << std::endl;
	if (key=='q' || key=='Q')
	{
		exit(0);
	}
	return false;
}


int main(int argc, char *argv[])
{

    std::cout<<"eigen version.:"<<EIGEN_WORLD_VERSION<<"."<<EIGEN_MAJOR_VERSION <<"."<< EIGEN_MINOR_VERSION<<"\n";

	//################################################################
	// Init the viewer
	igl::opengl::glfw::Viewer viewer;

	// Attach a menu plugin
	igl::opengl::glfw::imgui::ImGuiMenu menu;
	viewer.plugins.push_back(&menu);

	// menu variable Shared between two menus
	double doubleVariable = 0.1f; 

	// Add content to the default menu window via defining a Lambda expression with captures by reference([&])
	menu.callback_draw_viewer_menu = [&]()
	{
		// Draw parent menu content
		menu.draw_viewer_menu();

		// Add new group
		if (ImGui::CollapsingHeader("New Group", ImGuiTreeNodeFlags_DefaultOpen))
		{
			// Expose variable directly ...
			ImGui::InputDouble("double", &doubleVariable, 0, 0, "%.4f");

			// ... or using a custom callback
			static bool boolVariable = true;
			if (ImGui::Checkbox("bool", &boolVariable))
			{
				// do something
				std::cout << "boolVariable: " << std::boolalpha << boolVariable << std::endl;
			}

			// Expose an enumeration type
			enum Orientation { Up = 0, Down, Left, Right };
			static Orientation dir = Up;
			ImGui::Combo("Direction", (int *)(&dir), "Up\0Down\0Left\0Right\0\0");

			// We can also use a std::vector<std::string> defined dynamically
			static int num_choices = 3;
			static std::vector<std::string> choices;
			static int idx_choice = 0;
			if (ImGui::InputInt("Num letters", &num_choices))
			{
				num_choices = std::max(1, std::min(26, num_choices));
			}
			if (num_choices != (int)choices.size())
			{
				choices.resize(num_choices);
				for (int i = 0; i < num_choices; ++i)
					choices[i] = std::string(1, 'A' + i);
				if (idx_choice >= num_choices)
					idx_choice = num_choices - 1;
			}
			ImGui::Combo("Letter", &idx_choice, choices);

			// Add a button
			if (ImGui::Button("Print Hello", ImVec2(-1, 0)))
			{
				std::cout << "Hello\n";
			}
		}
	};

	// Add additional windows via defining a Lambda expression with captures by reference([&])
	menu.callback_draw_custom_window = [&]()
	{
		// Define next window position + size
		ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling(), 10), ImGuiSetCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(300, 450), ImGuiSetCond_FirstUseEver);
		ImGui::Begin( "MyProperties", nullptr, ImGuiWindowFlags_NoSavedSettings );
		
		// point size
		// [event handle] if value changed
		if (ImGui::InputFloat("point_size", &g_myctx.point_size))
		{
			std::cout << "point_size changed\n";
			viewer.data().point_size = g_myctx.point_size;
		}

		// line width
		// [event handle] if value changed
		if(ImGui::InputFloat("line_width", &g_myctx.line_width))
		{
			std::cout << "line_width changed\n";
			viewer.data().line_width = g_myctx.line_width;
		}

		//mode
		ImGui::SliderInt("Mode", &g_myctx.mode, 0,8);
        // select model
        ImGui::SliderInt("Models to load", &g_myctx.nModels, 2,1500);
        ImGui::SliderInt("Target model", &g_myctx.model, 1, g_myctx.maxModel);

        ImGui::Checkbox("Reduction", &g_myctx.reduction);
        ImGui::InputInt("Reduction size", &g_myctx.reductionSize, 1,500);

        //ImGui::SliderInt("# eigenvalues", &g_myctx.k_eig, 1, 10);

        ImGui::InputFloat("threshold", &g_myctx.threshold, 0.5,0.1);

        if(ImGui::SliderFloat("Scale0", &g_myctx.scaling0, -10,10)){
            g_myctx.reset_display(viewer);
        }
        if(ImGui::SliderFloat("Scale1", &g_myctx.scaling1, -10,10)){
            g_myctx.reset_display(viewer);
        }
        if(ImGui::SliderFloat("Scale2", &g_myctx.scaling2, -10,10)){
            g_myctx.reset_display(viewer);
        }
        if(ImGui::SliderFloat("Scale3", &g_myctx.scaling3, -10,10)){
            g_myctx.reset_display(viewer);
        }
        if(ImGui::SliderFloat("Scale4", &g_myctx.scaling4, -10,10)){
            g_myctx.reset_display(viewer);
        }
        if(ImGui::SliderFloat("Scale5", &g_myctx.scaling5, -10,10)){
            g_myctx.reset_display(viewer);
        }



		//mode-text
		if (g_myctx.mode == 1) { 
			ImGui::Text("mode: loading meshes ");
		}
        else if (g_myctx.mode == 2) {
            ImGui::Text("mode: show average shape ");
        }
        else if (g_myctx.mode == 3) {
            ImGui::Text("mode: PCA with the mean ");
        }
        else if (g_myctx.mode == 4){
            ImGui::Text("mode: upsample to original size");
        }
        else if (g_myctx.mode == 5) {
            ImGui::Text("mode: PCA with a target mesh");
        }
        else if (g_myctx.mode == 7) {
            ImGui::Text("mode: testing PCA on a single model ");
        }
        else if (g_myctx.mode == 8) {
            ImGui::Text("mode: testing with a single model ");
        }
        else {
            ImGui::Text("mode: ");
        }

        if (ImGui::Button("Apply")) {
        	std::cout << "Applying" << std::endl;
            g_myctx.reset_display(viewer);
        }

        if (g_myctx.model == 1) {
            ImGui::Text("model: ");
        }
        else{
            ImGui::Text("model: body ");
        }


        ImGui::End();
	};


	// registered a event handler
	viewer.callback_key_down = &key_down;

	g_myctx.reset_display(viewer);

	// Call GUI
	viewer.launch();

}


void get_example_mesh(std::string const meshname , Eigen::MatrixXd & V, Eigen::MatrixXi & F, Eigen::MatrixXd & VN)
{
    
    
    std::vector<const char *> cands{
        "../../data/",
        "../../../data/",
        "../../../../data/",
        "../../../../../data/" };
    
    bool found = false;
    for (const auto & val : cands)
    {
        if ( igl::file_exists(val+ meshname) )
        {
            std::cout << "loading example mesh from:" << val+ meshname << "\n";
            
            //if (igl::readOFF(val+ meshname, V,F)) {
            if (igl::readOBJ(val+ meshname, V,F)) {
                igl::per_vertex_normals(V, F, VN);
                found = 1;
                break;
            }
            else {
                std::cout << "file loading failed " << cands[0] + meshname << "\n";
            }
        }
    }
    
    if (!found) {
        std::cout << "cannot locate "<<cands[0]+ meshname <<"\n";
        exit(1);
    }
    
}
