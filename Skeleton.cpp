//=============================================================================================
// Mintaprogram: Z�ld h�romsz�g. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!!
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    :
// Neptun :
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

using std::vector;
typedef unsigned int uint;

// Dual numbers for automatic derivation
template <class T> struct Dnum
{
	// function value
	float f;
	// deivative
	T d;
	Dnum(float f0 = 0, T d0 = T(0))
	{
		f = f0;
		d = d0;
	}
	Dnum operator+ (Dnum r) { return Dnum(f + r.f, d + r.d); }
	Dnum operator- (Dnum r) { return Dnum(f - r.f, d - r.d); }
	Dnum operator* (Dnum r) { return Dnum(f * r.f, f * r.d + d * r.f);}
	Dnum operator/ (Dnum r) { return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);}
};

template <class T> Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f)*g.d); }
template <class T> Dnum<T> Sin(Dnum<T> g) { return Dnum<T>(sinf(g.f), cosf(g.f)*g.d); }
template <class T> Dnum<T> Cos(Dnum<T> g) { return Dnum<T>(cosf(g.f), -sinf(g.f)*g.d); }
template <class T> Dnum<T> Tan(Dnum<T> g) { return Sin(g) / Cos(g); }
template <class T> Dnum<T> Sinh(Dnum<T> g) { return Dnum<T> (sinh(g.f), cosh(g.f)*g.d);}
template <class T> Dnum<T> Cosh(Dnum<T> g) { return Dnum<T> (cosh(g.f), sinh(g.f)*g.d);}
template <class T> Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g);}
template <class T> Dnum<T> Log(Dnum<T> g) { return Dnum<T> (logf(g.f), g.d / g.f);}
template <class T> Dnum<T> Pow(Dnum<T> g, float n) { return Dnum<T> (powf(g.f, n), n * powf(g.f, n-1) * g.d);}

typedef Dnum<vec2> Dnum2;

const int tesselationLevel = 20;

struct Camera
{
	//extrinsic parameters
	vec3 wEye, wLookat, wVup;
	// interinsic parameters
	// fov : field of view, asp: aspect ratio, fp: front clipping, bp: back clipping
	float fov, asp, fp, bp;

public:
	Camera()
	{
		asp = (float) windowWidth/windowHeight;
		fov = 75.0f * (float) M_PI / 180.0f;
		fp = 1;
		bp = 20;
	}
	// view matrix
	mat4 V()
	{
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);

		return TranslateMatrix(wEye * (-1)) *
										mat4 (u.x,  v.x,  w.x,  0.0f,
													u.y,  v.y,  w.y,  0.0f,
													u.z,  v.z,  w.z,  0.0f,
													0.0f, 0.0f, 0.0f, 1.0f);
	}

	// projection matrix
	mat4 P()
	{
		// scale the viewing pyramid to 90 degrees
		float sy = 1/tan(fov/2);
		return mat4	(sy/asp,	0.0f,	0.0f,							0.0f,
								 0.0f,		sy,		0.0f,							0.0f,
							 	 0.0f,		0.0f,	-(fp+bp)/(bp-fp),	-1.0f,
							 	 0.0f,		0.0f,	-(2*fp*bp)/(bp-fp), 0.0f);
	}
};

struct OrthoCamera
{
	//extrinsic parameters
	vec3 wEye, wLookat, wVup;
	// interinsic parameters
	// fov : field of view, asp: aspect ratio, fp: front clipping, bp: back clipping
	float fov, asp, fp, bp;
	float left, right, top, bottom, near, far;

public:
	OrthoCamera()
	{
		asp = (float) windowWidth/windowHeight;
		fov = 75.0f * (float) M_PI / 180.0f;
		fp = 1;
		bp = 20;
	}
	// view matrix
	mat4 V()
	{
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);

		return TranslateMatrix(wEye * (-1)) *
										mat4 (u.x,  v.x,  w.x,  0.0f,
													u.y,  v.y,  w.y,  0.0f,
													u.z,  v.z,  w.z,  0.0f,
													0.0f, 0.0f, 0.0f, 1.0f);
	}

	// projection matrix
	mat4 P()
	{
		// scale the viewing pyramid to 90 degrees
		// float sy = 1/tan(fov/2);
		return mat4	(1.5f,	0.0f,	0.0f,				0.0f,
								 0.0f,		1.5f,		0.0f,	0.0f,
							 	 0.0f,		0.0f, -1.0f,	0.0f,
							 	 0.0f,		0.0f,	0.0f, 1.0f);
	}
};

struct Material
{
	vec3 kd, ks, ka;
	float shininess;
};

struct Light
{
	vec3 La, Le;
	vec4 wLightPos;
};

class CheckerBoardTexture : public Texture
{
public:
	CheckerBoardTexture(const int width, const int height) : Texture()
	{
		vector<vec4> image(width * height);
		const vec4 yellow (1,1,0,1);
		const vec4 blue(0,0,1,1);
		for (int x = 0; x < width; ++x)
		{
			for (int y = 0; y < height; ++y)
			{
				image[y * width + x] = (x & 1) ^ (y & 1) ? yellow : blue;
			}
		}
		create(width, height, image, GL_NEAREST);
	}
};

class BowlTexture : public Texture
{
public:
	BowlTexture(const int width, const int height, const uint numberCircles) : Texture()
	{
		// light and dark brown colors
		const vec4 lbrown(0.68f, 0.46f, 0.22f, 1.0f);
		const vec4 dbrown(0.28f, 0.16f, 0.03f, 1.0f);
		// initialize image to zero
		vector<vec4> image(width * height, dbrown);

		float t = 0.0f;
		float r = 0.0f;
		int xcord = 0;
		int ycord = 0;
		// the int casting to tranculate any deecimal point
		int maxrad = width < height ? width : height;
		maxrad = maxrad / 2;

		for (uint i = 0; i < numberCircles; ++i)
		{
			// t is between 0 and 1
			t = i / (numberCircles - 1.0f);
			// radius decrease from 1 to 1/numberCircles
			r = (float) maxrad * (1 - t) + 1.0f / numberCircles * t;
			// check for odd and even circle and assign to color (c)
			vec4 c = (i % 2 == 0) ? lbrown : dbrown;
			for (int x = maxrad; x < width ; ++x)
			{
				xcord = x - maxrad;
				for (int y = 0; y < height; ++y)
				{
					ycord = y - maxrad;
					if (xcord * xcord + ycord * ycord <= r * r)
					{
						image[y * width + x] = c;
						image[y * width + (maxrad - xcord)] = c;
						image[(maxrad - ycord) * width + x] = c;
						image[(maxrad - ycord) * width + (maxrad - xcord)] = c;
					}
					// image[y * width + x] = lbrown;
				}
			}
		}
		create(width, height, image, GL_NEAREST);
	}
};

class RedTexture : public Texture
{
public:
	RedTexture(const int width, const int height) : Texture()
	{
		// light and dark brown colors
		const vec4 color(1.0f, 0.0f, 0.0f, 1.0f);
		// initialize image to zero
		vector<vec4> image(width * height, color);
		create(width, height, image, GL_NEAREST);
	}
};

class GreenTexture : public Texture
{
public:
	GreenTexture(const int width, const int height) : Texture()
	{
		// light and dark brown colors
		const vec4 color(0.0f, 1.0f, 0.0f, 1.0f);
		// initialize image to zero
		vector<vec4> image(width * height, color);
		create(width, height, image, GL_NEAREST);
	}
};

class BlueTexture : public Texture
{
public:
	BlueTexture(const int width, const int height) : Texture()
	{
		// light and dark brown colors
		const vec4 color(0.0f, 0.0f, 1.0f, 1.0f);
		// initialize image to zero
		vector<vec4> image(width * height, color);
		create(width, height, image, GL_NEAREST);
	}
};

struct RenderState
{
	mat4					MVP, M, Minv, V, P;
	Material*			material;
	vector<Light>	lights;
	Texture *			texture;
	vec3 					wEye;
};

struct Shader : public GPUProgram
{
public:
	virtual void Bind(RenderState state) = 0;

	void setUniformMaterial(const Material &material, const std::string &name)
	{
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shininess, name + ".shininess");
	}

	void setUniformLight(const Light &light, const std::string &name)
	{
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}
};

class PhongShader : public Shader
{
	const char * vertexSource =
	R"(
		#version 330
		precision highp float;

		struct Light
		{
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4 MVP, M, Minv;
		uniform Light[8] lights;
		uniform int nLights;
		uniform vec3 wEye;

		layout(location = 0) in vec3 vtxPos;
		layout(location = 1) in vec3 vtxNorm;
		layout(location = 2) in vec2 vtxUV;

		out vec3 wNormal;
		out vec3 wView;
		out vec3 wLight[8];
		out vec2 texcoord;

		void main()
		{
			gl_Position = vec4(vtxPos, 1) * MVP;
			vec4 wPos = vec4(vtxPos, 1) * M;

			for (int i = 0; i < nLights; ++i) { wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w; }
			wView = wEye * wPos.w - wPos.xyz;
			wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
			texcoord = vtxUV;
		}

	)";

	const char * fragmentSource =
	R"(
		#version 330
		precision highp float;

		struct Light
		{
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material
		{
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;
		uniform int nLights;
		uniform sampler2D diffuseTexture;

		in vec3 wNormal;
		in vec3 wView;
		in vec3 wLight[8];
		in vec2 texcoord;

		out vec4 fragmentColor;

		void main()
		{
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView);
			if (dot(N,V) < 0) {N = -N;}
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;

			vec3 radiance = vec3(0,0,0);
			for (int i = 0; i < nLights; ++i)
			{
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0);
				float cosd = max(dot(N,H), 0);

				radiance += ka * lights[i].La + (kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";

public:
	PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }
	void Bind(RenderState state)
	{
		Use(); // make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniform(*state.texture, std::string("diffuseTexture"));
		setUniformMaterial(*state.material, "material");

		setUniform((int) state.lights.size(), "nLights");
		for (uint i = 0; i < state.lights.size(); ++i)
		{
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};

/*
struct Shader
{
	uint shaderProg;

	// vsSrc: vertex shader source code
	void Create(const char * vsSrc, const char * fsSrc, const char * fsOutputName)
	{
		uint vs = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vs, 1, &vsSrc, NULL);
		glCompileShader(vs);
		uint fs = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fs, 1, &fsSrc, NULL);
		glCompileShader(fs);
		shaderProgram = glCreateProgram();
		glAttachShader(shaderProg, vs);
		glAttachShader(shaderProg, fs);

		glBindFragDataLocation(shaderProg, 0, fsOutputName);
		glLinkProgram(shaderProg);
	}
	virtual void Bind (RenderState &state) { glUseProgram(shaderProg);}
};
*/


// The general base class of triangle mesh is Geomet ry
struct Geometry
{
protected:
	uint vao;
	uint vbo;
public:
	Geometry()
	{
		// triangle mesh is copied to GPU and assigned to vertex array object (vao)
		glGenVertexArrays (1, &vao);
		glBindVertexArray (vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}
	virtual void Draw () = 0;
	~ Geometry()
	{
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

struct VertexData
{
	vec3 pos,norm;
	vec2 tex;
};

// For parametric surface
struct ParamSurface : Geometry
{
	uint nVtxPerStrip, nStrips;

public:
	ParamSurface() { nVtxPerStrip = 0; nStrips = 0;}
	// this function is made for each parametric equation
	// virtual void eval (float u, float v, vec3 &pos, vec3 &norm) = 0;

	// get U and V and return X, Y, Z
	virtual void eval (Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;

	// creation of the surface data on GPU happens in this function
	void create (int N, int M);

	VertexData GenVertexData (float u, float v)
	{
		VertexData vtxData;
		Dnum2 X,Y,Z;
		Dnum2 U(u, vec2(1,0));
		Dnum2 V(v, vec2(0,1));
		// eval (u, v, vtxData.pos, vtxData.norm);
		eval(U, V, X, Y, Z);
		vec3 drdU(X.d.x, Y.d.x, Z.d.x);
		vec3 drdV(X.d.y, Y.d.y, Z.d.y);
		// take the function value
		vtxData.pos = vec3(X.f, Y.f, Z.f);
		vtxData.norm = cross(drdU, drdV);
		vtxData.tex = vec2 (u, v);

		return vtxData;
	}

	// Draw the mesh as a combination of triangle strips
	void Draw()
	{
		glBindVertexArray(vao);
		for (uint i = 0; i < nStrips; ++i)
		{ glDrawArrays (GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip); }
	}
};

void ParamSurface::create(int N = tesselationLevel, int M = tesselationLevel)
{
	nVtxPerStrip = (M + 1) * 2;
	nStrips = N;
	vector<VertexData> vtxData; // CPU-n

	// Add vertices in required strip order
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j <= M; ++j)
		{
			vtxData.push_back(GenVertexData((float) j/M, (float) i/N));
			vtxData.push_back(GenVertexData((float) j/M, (float) (i + 1)/N));
		}
	}

	// uint vbo; // GPU-n
	// glGenBuffers (1, &vbo);
	// glBindBuffer (GL_ARRAY_BUFFER, vbo);
	glBufferData (GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
	// input register number 0 will get the position
	glEnableVertexAttribArray(0); // Attrubute array 0 = POSITION
	// input register number 1 will get the normal
	glEnableVertexAttribArray(1); // Attrubute array 1 = NORMAL
	// inout register number 2 will get the UV pair of a particular point
	glEnableVertexAttribArray(2); // Attrubute array 2 = UV
	// attribite array, components per attribite, component type, normalize?, stride, offset
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*) offsetof(VertexData, pos));
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*) offsetof(VertexData, norm));
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*) offsetof(VertexData, tex));
}

class Sphere : public ParamSurface
{
public:
	Sphere() { create(); }
	void eval (Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z)
	{
		// transform the unit interval [0,1] to 2pi interval [0, 2pi]
		U = U * 2.0f * (float) M_PI;
		// transform the unit interval [0,1] to pi interval [0, pi]
		V = V * (float) M_PI;
		X = Cos(U) * Sin(V);
		Y = Sin(U) * Sin(V);
		Z = Cos(V);
	}
};

class Bowl : public ParamSurface
{
public:
	Bowl() { create(); }
	void eval (Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z)
	{
		float factor = 2;
		U = U * factor;
		U = U - factor / 2;
		V = V * factor;
		V = V - factor / 2;
		X = U;
		Y = V;
		Z = Cosh(Pow(U,2) + Pow(V,2));
	}
};

/*
class Sphere : public ParamSurface
{
	vec3 center;
	float radius;
public:
	void eval (float u, float v, vec3 &pos, vec3 &normal)
	{
		float U = u * 2 * M_PI;
		float V = v * M_PI;
		normal = vec3 (cos(U) * sin(V), sin(U) * sin(V), cos(V));
		pos = normal * radius + center;
	}
};
*/

class Object
{
	Shader * shader;
	Material * material;
	Texture * texture;
	Geometry * geometry;
public:
	vec3 scale, pos, rotAxis;
	float rotAngle;
	Object (Shader * _shader, Material * _material, Texture * _texture, Geometry * _geometry) :
	scale(vec3(1,1,1)), pos(vec3(0,0,0)), rotAxis(vec3(0,0,1)), rotAngle(0)
	{
		shader = _shader;
		texture = _texture;
		material = _material;
		geometry = _geometry;
	}

	virtual void SetModelingTransform (mat4& M, mat4& Minv)
	{
		M = ScaleMatrix(scale) * RotationMatrix(rotAngle, rotAxis) * TranslateMatrix(pos);
		Minv = TranslateMatrix(-pos) * RotationMatrix(-rotAngle, rotAxis) * ScaleMatrix(vec3(1/scale.x, 1/scale.y, 1/scale.z));
	}

	void Draw(RenderState state)
	{
		mat4 M, Minv;
		SetModelingTransform(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		shader -> Bind(state);
		geometry -> Draw();
	}
	/*
	void Draw (RenderState state)
	{
		state.M = Scale(scale.x, scale.y, scale.z) *
		 					Rotate(rotAngle, rotAxis.x, rotAxis.y, rotAxis.z) *
							Translate(pos.x, pos.y, pos.z);

		state.Minv = Translate(-pos.x, -pos.y, -pos.z) *
								Rotate(-rotAngle, rotAxis.x, rotAxis.y, rotAxis.z) *
								Scale(1/scale.x, 1/scale.y, 1/scale.z);
		state.material = material;
		state.texture = texture;
		shader -> Bind(state);
		geometry -> Draw();
	}
	*/
	// virtual void Animate(float dt) {}
	virtual void Animate(float tstart, float tend) { rotAngle = 0.8f * tend; }
};

class Scene
{
	OrthoCamera camera;
	vector<Object *> objects;
	vector<Light> lights;
public:
	void Build()
	{
		// Shader
		Shader * phongShader = new PhongShader();

		// Materials
		Material * material0 = new Material;
		material0 -> kd = vec3(0.6f, 0.4f, 0.2f);
		material0 -> ks = vec3(4,4,4);
		material0 -> ka = vec3(0.1f, 0.1f, 0.1f);
		material0 -> shininess = 100;

		Material * material1 = new Material;
		material1 -> kd = vec3(0.8f, 0.6f, 0.4f);
		material1 -> ks = vec3(0.3f, 0.3f, 0.3f);
		material1 -> ka = vec3(0.2f, 0.2f, 0.2f);
		material1 -> shininess = 30;

		// Textures
		Texture * texture4x8 = new CheckerBoardTexture(4,8);
		Texture * texture15x20 = new CheckerBoardTexture(15,20);
		Texture * bowl50x50 = new BowlTexture(512,512,10);
		Texture * red4x4 = new RedTexture(4,4);
		Texture * green4x4 = new GreenTexture(4,4);
		Texture * blue4x4 = new BlueTexture(4,4);

		// Geometries
		Geometry * sphere = new Sphere();
		Geometry * bowl = new Bowl();

		// Objects
		Object * sphereObject1 = new Object(phongShader, material0, red4x4, sphere);
		sphereObject1 -> pos = vec3(-0.6, -0.6, 1.5);
		sphereObject1 -> scale = vec3(0.05f, 0.05f, 0.05f);
		objects.push_back(sphereObject1);

		Object * sphereObject2 = new Object(phongShader, material0, green4x4, sphere);
		sphereObject2 -> pos = vec3(0, 0, 1);
		sphereObject2 -> scale = vec3(0.05f, 0.05f, 0.05f);
		objects.push_back(sphereObject2);

		Object * bowlObject = new Object(phongShader, material0, bowl50x50, bowl);
		bowlObject -> pos = vec3(0,0,0);
		// bowlObject -> rotAxis = vec3(0,0,1);
		// bowlObject -> rotAngle = 1.570;
		objects.push_back(bowlObject);

		// Camera
		camera.wEye = vec3(0,0,2);
		camera.wLookat = vec3(0,0,0);
		camera.wVup = vec3(0, 1, 0);

		// Lights
		// directional light source
		float am = 0.3f;
		float pt = 0.8f;
		lights.resize(2);
		lights[0].wLightPos = vec4(5,5,10,0);
		lights[0].La = vec3(am, am, am);
		lights[0].Le = vec3(pt, pt, pt);
		// lights[0].La = vec3(0.1f, 0.1f, 1);
		// lights[0].Le = vec3(wl, wl, wl);

		// directional light source
		lights[1].wLightPos = vec4(-5,-5,10,0);
		lights[1].La = vec3(am, am, am);
		lights[1].Le = vec3(pt, pt, pt);

		// directional light source
		// lights[2].wLightPos = vec4(-5, 5, 5, 0);
		// lights[2].La = vec3(0.1f, 0.1f, 0.1f);
		// lights[2].Le = vec3(0, 0, 3);
	}
	void Render()
	{
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.lights = lights;
		for (Object * obj : objects) {obj -> Draw(state);}
	}

	/*
	void Animate(float dt)
	{
		for (Object * obj : objects) { obj -> Animate(dt); }
	}
	*/

	void Animate(float tstart, float tend)
	{
		// for (Object * obj : objects) { obj -> Animate(tstart, tend); }
		lights[0].wLightPos.y -= 0.01f;
		lights[1].wLightPos.y += 0.01f;
	}
};

/*
// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char *vertexSource = R"(
	#version 330				// Shader 3.3
	uniform mat4 M,  Minv, MVP;
	// position of light source
	uniform vec4 wLiPos;
	// position of eye
	uniform vec3 wEye;

	// moodel position
	layout (location = 0) in vec3 vtxPos;
	// model normal
	layout (location = 1) in vec3 vtxNorm;

	out vec3 wNormal;
	out vec3 wView;
	out vec3 wLight;

	// vertex shader
	void main()
	{
		// transform to normalized device coordinates
		gl_Position = vec4 (vtxPos, 1) * MVP;
		// transform to world coordinates
		vec4 wPos = vec4 (vtxPos, 1) * M;
		// computation of illumination
		wLight = wLipos.xyz * wPos.w - wPos.xyz * wLiPos.w;
		// computation of view direction
		wView = wEye * wPos.w - wPos.xyz;
		wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
	}
)";
*/

/*
// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	// diffuse, specular, and ambient
	uniform vec3 kd, ks, ka;
	// ambient and point source
	uniform vec3 La, Le;
	// shininess for specular ref

	in vec3 wNormal;
	in vec3 wView;
	in vec3 wLight;

	out vec4 fragmentColor;

	void main() {
		vec3 N = normalize(wNormal);
		vec3 V = normalize(wView);
		vec3 L = normalize(wLight);
		vec3 H = normalize(L + V);
		float cost = max (dot(N,L), 0);
		float cosd = max (dot(N,H), 0);
		vec3 color = ka * La + (kd * cost + ks * pow(cosd, shine)) * Le;
		fragmentColor = vec4(color,1);
	}
)";
*/

Scene scene;

// Initialization, create an OpenGL context
void onInitialization()
{
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();

}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.5f, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear frame buffer
	scene.Render();
	glutSwapBuffers();
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
void onMouseMotion(int pX, int pY) {
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
void onMouse(int button, int state, int pX, int pY) {
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = (char *) "pressed"; break;
	case GLUT_UP:   buttonStat = (char *) "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:
	   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);
		 break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float tstart = 0.5f;
	float tend = 0.6f;
	scene.Animate(tstart, tend);
	glutPostRedisplay();
}
