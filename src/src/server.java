import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import com.sun.net.httpserver.HttpHandler;

import java.io.*;
import java.net.InetSocketAddress;
import main.Node;
import main.Utils;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.rel.RelNode;
import org.apache.commons.io.IOUtils;
import java.text.DateFormat;
import java.util.Date;
import main.Rewriter;
import com.google.gson.JsonObject;
import verify.verifyrelnode;

public class server {
    public static void main(String[] arg) throws Exception {
        HttpServer server = HttpServer.create(new InetSocketAddress(6336), 0);
        server.createContext("/rewrite", new RequestHandler());
        server.createContext("/test", new GetHandler());
        server.createContext("/parser", new ParserHandler());
        server.createContext("/eq", new EQHandler());
        server.start();
    }

    static class GetHandler implements HttpHandler{
        public void handle(HttpExchange exchange) throws IOException {
            try{
                //获得表单提交数据(post)
                String postString = IOUtils.toString(exchange.getRequestBody());
                System.out.println("postString：" + postString);
                JSONObject postInfo = JSONObject.parseObject(postString);
                System.out.println("请求参数：" + postInfo);
                String sql = postInfo.getString("sql");
                System.out.println("sql：" + sql);
            } catch (Exception e) {
                System.out.println("处理失败");
            } finally {
                OutputStream os = exchange.getResponseBody();
                exchange.sendResponseHeaders(200,0);
                os.write("success".getBytes());
                os.close();
            }
        }
    }

    static class RequestHandler implements HttpHandler{
        public void handle(HttpExchange exchange) throws IOException {
            JSONObject responseJson = new JSONObject();

            String logString = DateFormat.getInstance().format(new Date()).toString();
            logString += "\n";

            try{
                //获得表单提交数据(post)
                String postString = IOUtils.toString(exchange.getRequestBody());
                logString += postString;
                logString += "\n";
                JSONObject postInfo = JSONObject.parseObject(postString);
                String sql = postInfo.getString("sql");
                String schemaJson = postInfo.getString("schema");

                System.out.println("请求sql：" + sql);
                if (sql == null) {
                    responseJson.put("status", false);
                    responseJson.put("message", "Please enter Sql");
                }else {
                    sql = sql.replace(";", "");
                    //DB Config
                    JSONArray jobj;
                    if (schemaJson == null) {
                        String path = System.getProperty("user.dir");
                        jobj = Utils.readJsonFile(path+"/src/main/schema.json");
                    }else {
                        jobj = JSON.parseArray(schemaJson);
                    }
                    Rewriter rewriter = new Rewriter(jobj);
                    RelNode relNode = rewriter.SQL2RA(sql);
                    double origin_cost = rewriter.getCostRecordFromRelNode(relNode);
                    Node resultNode = new Node(sql, relNode, (float) origin_cost,rewriter, (float) 0.1,null,"original query");
                    Node res = resultNode.UTCSEARCH(20, resultNode,1);
                    JSONObject dataJson = new JSONObject();
                    JSONObject treeJson = Utils.generate_json(resultNode);
                    dataJson.put("origin_cost", String.format("%.4f",origin_cost));
                    dataJson.put("origin_sql", sql);
                    dataJson.put("origin_sql_node", RelOptUtil.toString(relNode));
                    dataJson.put("rewritten_cost", String.format("%.4f",rewriter.getCostRecordFromRelNode(res.state_rel)));
                    dataJson.put("rewritten_sql", res.state);
                    RelNode rewrittenRelNode = rewriter.SQL2RA(res.state);
                    dataJson.put("rewritten_sql_node", RelOptUtil.toString(rewrittenRelNode));
                    dataJson.put("is_rewritten", !res.state.equalsIgnoreCase(sql));
                    dataJson.put("treeJson", treeJson);
                    responseJson.put("status", true);
                    responseJson.put("message", "SUCCESS");
                    responseJson.put("data", dataJson);
                    logString += responseJson.toJSONString();
                    logString += "\n";
                    logString += DateFormat.getInstance().format(new Date()).toString();
                    logString += "\n";
                    logString += "====================================================\n\n";
                }
            } catch (Exception e) {
                responseJson.put("status", false);
                responseJson.put("message", "Get Error");
                logString += e.toString();
                logString += "\n";
                logString += "====================================================\n\n";
            } finally {
                OutputStream os = exchange.getResponseBody();
                exchange.sendResponseHeaders(200,0);
                os.write(responseJson.toJSONString().getBytes());
                os.close();
                try{
                    FileOutputStream o = null;
                    File file = new File("request.txt");
                    if(!file.exists()){
                        file.createNewFile();
                    }
                    byte[] buff = new byte[]{};
                    buff=logString.getBytes();
                    o=new FileOutputStream(file,true);
                    o.write(buff);
                    o.flush();
                    o.close();
                }catch(Exception e){
                    e.printStackTrace();
                }
            }
        }
    }

    static class ParserHandler implements HttpHandler{
        public void handle(HttpExchange exchange) throws IOException {
            JSONObject responseJson = new JSONObject();
            try{
                //获得表单提交数据(post)
                String postString = IOUtils.toString(exchange.getRequestBody());
                JSONObject postInfo = JSONObject.parseObject(postString);
                String sql = postInfo.getString("sql");
                String schemaJson = postInfo.getString("schema");

                System.out.println("请求sql：" + sql);
                if (sql == null) {
                    responseJson.put("status", false);
                    responseJson.put("message", "Please enter Sql");
                }else {
                    sql = sql.replace(";", "");
                    //DB Config
                    JSONArray jobj;
                    if (schemaJson == null) {
                        String path = System.getProperty("user.dir");
                        jobj = Utils.readJsonFile(path+"/src/main/schema.json");
                    }else {
                        jobj = JSON.parseArray(schemaJson);
                    }
                    Rewriter rewriter = new Rewriter(jobj);
                    RelNode relNode = rewriter.SQL2RA(sql);
                    JSONObject dataJson = new JSONObject();
                    dataJson.put("res_node", RelOptUtil.toString(relNode));
                    responseJson.put("status", true);
                    responseJson.put("message", "SUCCESS");
                    responseJson.put("data", dataJson);
                }
            } catch (Exception e) {
                responseJson.put("status", false);
                responseJson.put("message", "Get Error");
            } finally {
                OutputStream os = exchange.getResponseBody();
                exchange.sendResponseHeaders(200,0);
                os.write(responseJson.toJSONString().getBytes());
                os.close();
            }
        }
    }

    static class EQHandler implements HttpHandler{
        public void handle(HttpExchange exchange) throws IOException {
            JSONObject responseJson = new JSONObject();
            try{
                //获得表单提交数据(post)
                String postString = IOUtils.toString(exchange.getRequestBody());
                JSONObject postInfo = JSONObject.parseObject(postString);
                String sql1 = postInfo.getString("sql1");
                String sql2 = postInfo.getString("sql2");
                String schemaJson = postInfo.getString("schema");

                if (sql1 == null || sql2 == null) {
                    responseJson.put("status", false);
                    responseJson.put("message", "Please enter Sql");
                }else {
                    sql1 = sql1.replace(";", "");
                    sql2 = sql2.replace(";", "");
                    //DB Config
                    JSONArray jobj;
                    if (schemaJson == null) {
                        String path = System.getProperty("user.dir");
                        jobj = Utils.readJsonFile(path+"/src/main/schema.json");
                    }else {
                        jobj = JSON.parseArray(schemaJson);
                    }
                    Rewriter rewriter = new Rewriter(jobj);
                    RelNode relNode1 = rewriter.SQL2RA(sql1);
                    RelNode relNode2 = rewriter.SQL2RA(sql2);

                    System.out.println("===============开始执行=========");
                    JsonObject  verifyJson = verifyrelnode.verifyrelnode(relNode1, relNode2, sql1, sql2);
                    System.out.println("verifyJson:" + verifyJson);
                    responseJson.put("status", true);
                    responseJson.put("message", "SUCCESS");
                    responseJson.put("data", "1231231231231");
                }
            } catch (Exception e) {
                responseJson.put("status", false);
                responseJson.put("message", "Get Error");
            } finally {
                OutputStream os = exchange.getResponseBody();
                exchange.sendResponseHeaders(200,0);
                os.write(responseJson.toJSONString().getBytes());
                os.close();
            }
        }
    }

}
