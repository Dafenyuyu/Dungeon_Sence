Shader "Unlit/blend"
{
    Properties
    {
        _Control ("Control(RGBA)", 2D) = "white" {}

        _Splat0("Layer1(RGBA)" , 2D) = "white"{}
        _BumpSplat0("Layer 1 Normal(Bump)",2D) = "Bump"{}
        _Splat1("Layer2(RGBA)" , 2D) = "white"{}
        _BumpSplat1("Layer 2 Normal(Bump)",2D) = "Bump"{}
        _Splat2("Layer3(RGBA)" , 2D) = "white"{}
        _BumpSplat2("Layer 3 Normal(Bump)",2D) = "Bump"{}
        _Splat3("Layer4(RGBA)" , 2D) = "white"{}
        _BumpSplat3("Layer 4 Normal(Bump)",2D) = "Bump"{}

        _Gloss("Gloss",Range(0,1)) = 0.8

        _Specular("Specularcolor" , Color) = (1,1,1,1) 

        _weight("blend weight" , Range(0.001 , 1)) = 0.2
    }
    SubShader
    {
        Pass
        {
            Tags { "LightMode"="ForwardBase" }
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #include "Lighting.cginc"
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            sampler2D _Control;
            float4 _Control_ST;
            sampler2D _Splat0,_Splat1,_Splat2,_Splat3;
            float4 _Splat0_ST,_Splat1_ST,_Splat2_ST,_Splat3_ST;
            half _weight;
            sampler2D _BumpSplat0,_BumpSplat2,_BumpSplat3,_BumpSplat1;
            half _Gloss;
            half4 _Specular;

            // float4 _BumpSplat0_ST,_BumpSplat2_ST,_BumpSplat3_ST,_BumpSplat1_ST;

            //从模型拿到数据    （这一段结构体可以用appdata_full来代替）
            // struct VertexInput
            // {
            //     float4 vertex : POSITION;
            //     float2 uv : TEXCOORD0;
            // };

            struct VertexOutput
            {
                float4 vertex : SV_POSITION;
                float2 uv : TEXCOORD0;

                //用两个float4来代替四个float2 
                float4 uv1 : TEXCOORD2;
                float4 uv2 : TEXCOORD3;
                float4 TtoW0: TEXCOORD4;
                float4 TtoW1: TEXCOORD5;
                float4 TtoW2: TEXCOORD6;

                float3 worldNormalDir : COLOR0;

                UNITY_FOG_COORDS(1)

            };

            VertexOutput vert (appdata_full v)
            {
                VertexOutput o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.texcoord, _Control);

                o.uv1.xy = TRANSFORM_TEX(v.texcoord, _Splat0);
                o.uv1.zw = TRANSFORM_TEX(v.texcoord, _Splat1);
                o.uv2.xy = TRANSFORM_TEX(v.texcoord, _Splat2);
                o.uv2.zw = TRANSFORM_TEX(v.texcoord, _Splat3);

                // 世界空间法线
                float3 worldNormal = mul((float3x3)unity_ObjectToWorld,v.normal);
                // 世界空间顶点
                float3 worldPos = mul(unity_ObjectToWorld,v.vertex).xyz;
                // 世界空间切线
                fixed3 worldTangent = UnityObjectToWorldDir(v.tangent.xyz);
                // 世界空间副切线
                fixed3 worldBinormal = cross(worldNormal , worldTangent) * v.tangent.w;

                // 把这些信息存储到TtoW里面
                o.TtoW0 = float4(worldTangent.x , worldBinormal.x , worldNormal.x , worldPos.x);
                o.TtoW1 = float4(worldTangent.y , worldBinormal.y , worldNormal.y , worldPos.y);
                o.TtoW2 = float4(worldTangent.z , worldBinormal.z , worldNormal.z , worldPos.z);

                UNITY_TRANSFER_FOG(o,o.vertex);
                return o;
            }

            // 做一个高度混合函数
            inline half4 Blend(half4 depth1 , half4 depth2 , half4 depth3 , half4 depth4 , fixed4 control){
                half4 blend;
                
                // 获取混合的高度（这一步通过遮罩划范围）
                blend.r = depth1 * control.r;
                blend.g = depth2 * control.g;
                blend.b = depth3 * control.b;
                blend.a = depth4 * control.a;

                // 所有的高度取最大值
                half ma = max(blend.r , max(blend.g , max(blend.b , blend.a)));

                // 用原来的高度混合减去 权重偏移值的高度混合 然后乘以一个遮罩防止溢出
                blend =max(blend-ma  + _weight , 0)*control;

                // 最终做了一个标准化
                return blend / (blend.r + blend.g + blend.b + blend.a);

            }

            // 法线计算
            inline half3 NormalCompute(float4 normal , float4 Tw1 , float4 Tw2 , float4 Tw3){
                
                fixed3 WorldSpaceNormal;
                fixed3 tangentNormal;
                tangentNormal = UnpackNormal(normal);
                tangentNormal.z = sqrt(1 - saturate(dot(tangentNormal.xy , tangentNormal.xy)));
                WorldSpaceNormal = normalize(half3(dot(Tw1.xyz,tangentNormal) , dot(Tw2.xyz,tangentNormal) , dot(Tw3.xyz,tangentNormal)));
                return WorldSpaceNormal;

            }

            // inline half Norma Computer1(float4 normal , float4 Tw1 , float4 Tw2 , float4 Tw3){
            //     fixed3 WorldSpaceNormal;
            //     fixed3 tangentNormal;
            //     tangentNormal = UnpackNormal(normal);

            // }

            fixed4 frag (VertexOutput i) : SV_Target
            {
                //混合贴图
                fixed4 control = tex2D(_Control, i.uv);
                 
                fixed4 Lay1 = tex2D(_Splat0 , i.uv1.xy);
                fixed4 normal1 = tex2D(_BumpSplat0,i.uv1.xy);

                fixed4 Lay2 = tex2D(_Splat1 , i.uv1.zw);
                fixed4 normal2 = tex2D(_BumpSplat1 ,i.uv1.zw);

                fixed4 Lay3 = tex2D(_Splat2 , i.uv2.xy);
                fixed4 normal3 = tex2D(_BumpSplat2 ,i.uv2.xy);

                fixed4 Lay4 = tex2D(_Splat3 , i.uv2.zw);
                fixed4 normal4 = tex2D(_BumpSplat2 ,i.uv2.zw);

                // 世界空间顶点
                float3 worldPos = float3(i.TtoW0.w , i.TtoW1.w , i.TtoW2.w);

                // 世界空间灯光方向
                fixed3 worldLightDir = normalize(UnityWorldSpaceLightDir(worldPos));

                // 世界空间视口方向
                fixed3 worldViewDir = normalize(UnityWorldSpaceViewDir(worldPos));

                // 半角向量 
                fixed3 halfDir = normalize(worldViewDir + worldLightDir);

                // 获取unity环境光
                fixed4 ambient = UNITY_LIGHTMODEL_AMBIENT;

                // 通过混合算法得到混合的多通道
                half4 blend = Blend (Lay1.a , Lay2.a , Lay3 .a , Lay4 .a , control);

                // 混合法线合并
                fixed4 MergeNormal = normal1*blend.r + normal2*blend.g + normal3*blend.b + normal4*blend.a;

                // 贴图切线法转换导世界空间法线
                fixed3 worldNormal = NormalCompute(MergeNormal , i.TtoW0 ,i.TtoW1 , i.TtoW2);

                // 兰伯特光照
                fixed4 diffusecolor =_LightColor0 * max(0,dot(worldNormal,worldLightDir)) ;

                // 高光计算
                fixed4 Specularcolor = pow(max(0 , dot(worldNormal , halfDir)),_Gloss*1024)*_Specular;

                // 创建一个half4数据 存储颜色
                half4 Albedo;
                // 让平铺纹理颜色和高度混合通道混合
                Albedo = Lay1 * blend.r + Lay2 * blend.g + Lay3 * blend.b + Lay4 * blend.a;

                Albedo = Albedo*diffusecolor+Specularcolor+ambient;


                // apply fog
                UNITY_APPLY_FOG(i.fogCoord, col);
                return Albedo;
            }
            ENDCG
        }
    }
}
